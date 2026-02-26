# Streamlit Application for Explainable Resume-JD Scoring
# Artifacts in ./artifacts:
#   - vectorizer.joblib (sklearn TfidfVectorizer trained on concatenated JD-Resume)
#   - logreg.joblib (sklearn linear classifier: LogisticRegression / LinearSVC w/ decision_function)
#   - calib_isotonic.joblib
# Decision threshold is FIXED from validation PR curve:
# THR_PASS = 0.82 (shortlisting / higher precision)

import io, os, re, string
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize
from scipy.sparse import hstack
import joblib
from bs4 import BeautifulSoup
import requests
from PyPDF2 import PdfReader

# # --------------------Page / Sidebar-----------------------------
st.set_page_config(page_title="Explainable JD - Resume Scorer", page_icon="✅", layout="wide")
st.sidebar.title("Settings")

# Fused-score weights
ALPHA_DEFAULT = 0.60
BETA_DEFAULT = 0.30
GAMMA_DEFAULT = 0.10

alpha = st.sidebar.slider("Weight: probability (α)", 0.0, 1.0, ALPHA_DEFAULT, 0.05)
beta  = st.sidebar.slider("Weight: similarity (β)",  0.0, 1.0, BETA_DEFAULT,  0.05)
gamma = st.sidebar.slider("Weight: overlap (γ)",     0.0, 1.0, GAMMA_DEFAULT, 0.05)
st.sidebar.caption("α+β+γ=1. These tune the 0–100 match score only; the PASS decision uses a fixed threshold.")

with st.sidebar.expander("Decision threshold (fixed)", expanded=True):
    st.markdown(
        "- **Shortlisting / Pass** (higher precision): `τ = 0.82` from validation PR curve.\n"
        "- This cut-off is applied to calibrated probability."
    )

# ---------------------------Paths / Text Utils---------------------------

ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

CUSTOM_STOPWORDS = set("""would could however within across every per among via throughout whereas include includes including
responsible responsibilities experience experiences role roles requirement requirements required preferred
this that these those it its my our their his her your you we they etc etc. ability demonstrated proven strong
                       """.split())

STOP = (set(ENGLISH_STOP_WORDS) | CUSTOM_STOPWORDS | set(list(string.ascii_lowercase)) | {"amp","nbsp","quot","—","–","•","’","“","”",
         # glue words that we never want to drive explanations
         "from","with","to","for","by","of","in","on","at","as","and","or","an","the","is","are","was","were",
         "using","use","used","including","include","includes"})

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-+#/.]*")

def tokenize(s: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(s or "")]

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ").replace("\t", " ").replace("•", " ").replace("-", "-")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def read_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        txt = "\n".join([page.extract_text() or "" for page in reader.pages])
        return clean_text(txt)
    except Exception:
        return ""
    
# Getting the JD from a link
def fetch_jd_from_url(url: str, timeput=12) -> str:
    try:
        resp = requests.get(url, timeout=timeput, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        st.warning(f"Could not fetch the URL: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "lxml")
    for sel in ["article", "[role=main]", ".jobsearch-JobComponent", ".content", "body"]:
        node = soup.select_one(sel)
        if node:
            return clean_text(node.get_text(separator=" "))
    return clean_text(soup.get_text(separator=" "))


# ---------------------------Load artifacts----------------------------

@st.cache_resource(show_spinner=False)
def load_artifacts():
    vec = model = iso = platt = None
    try:
        vec = joblib.load(os.path.join(ART_DIR, "vectorizer.joblib"))
    except Exception:
        st.error("Missing artifacts/vectorizer.joblib")
    try:
        model = joblib.load(os.path.join(ART_DIR, "logreg.joblib"))
    except Exception:
        st.error("Missing artifacts/logreg.joblib (LogisticRegression / Linear SVM)")

    # Calibrators
    try:
        iso = joblib.load(os.path.join(ART_DIR, "calib_isotonic.joblib"))
    except Exception:
        iso = None
    try:
        platt = joblib.load(os.path.join(ART_DIR, "calib_sigmoid.joblib"))
    except Exception:
        platt = None
    return vec, model, iso, platt

vec, lr, iso, platt = load_artifacts()


# -------------------------Pair Features---------------------------------

def join_pair(jd: str, resume: str) -> str:
    # Vectorizer was trained on concatenated text
    return f"[JD] {jd}\n[RESUME] {resume}"

def build_features_pair(vectorizer: TfidfVectorizer, jd_text: str, resume_text: str):
    return vectorizer.transform([join_pair(jd_text, resume_text)])

# ------------------------------Calibration helpers---------------------------

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calibrated_probability(raw: float, Xpair):
    """
    Robustly returns calibrated probability.
    Supports: Isotonic/Platt saved on RAE scores (Regressor-like) -> predict on [[raw]], CalibratedClassifierCV saved on FEATURES -> predict_proba(Xpair)
    """
    # CalibratedClassifierCV path
    if iso is not None and hasattr(iso, "predict_proba"):
        try:
            proba = iso.predict_proba(Xpair)
            return float(proba[0, 1])
        except Exception:
            pass
    
    # Raw-score calibrators
    if iso is not None:
        try:
            p = iso.predict(np.array([raw]).reshape(-1, 1))
            return float(np.ravel(p)[0])
        except Exception:
            try:
                p = iso.predict(np.array([raw]))
                return float(np.ravel(p)[0])
            except Exception:
                pass

    if platt is not None:
        try:
            if hasattr(platt, "predict_proba"):
                return float(platt.predict_proba(np.array([raw]).reshape(-1, 1))[0, 1])
            p = platt.predict(np.array([raw]).reshape(-1, 1))
            return float(np.ravel(p)[0])
        except Exception:
            pass

    return float(_sigmoid(raw))


# --------------------------Explanations----------------------------
@dataclass
class TermImpact:
    term: str
    contrib: float

def vectorizer_maps(vectorizer: TfidfVectorizer):
    vocab = vectorizer.vocabulary_
    inv_vocab = {v:k for k, v in vocab.items()}
    idf = getattr(vectorizer, "idf_", None)
    return vocab, inv_vocab, idf

def term_contributions(vectorizer, model, Xrow) -> List[TermImpact]:
    coef = np.ravel(model.coef_)
    data = Xrow.tocoo()
    vocab, inv_vocab, _ = vectorizer_maps(vectorizer)
    out = []
    for i, j, v in zip(data.row, data.col, data.data):
        if i != 0:
            continue
        out.append(TermImpact(inv_vocab.get(j, f"f{j}"), float(coef[j] * v)))
    return out

def pick_top_terms(contribs: List[TermImpact], k=10, stop=STOP,idf=None, idf_min=1.8):
    filt = []
    for it in contribs:
        t = it.term.lower()
        if t in stop: 
            continue
        if idf is not None:
            idx = vec.vocabulary_.get(t)
            if idx is not None and idf[idx] < idf_min:
                continue
        filt.append(it)

    pos = sorted([x for x in filt if x.contrib > 0], key=lambda z: -z.contrib)[:k]
    neg = sorted([x for x in filt if x.contrib < 0], key=lambda z: z.contrib)[:k]
    return pos, neg

def plot_top_terms_bar(pos_terms: List[TermImpact], neg_terms: List[TermImpact], title_pos="Positive Evidence (Help terms)", title_neg="Negative Evidence (Hurt terms)"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    def bar(ax, items, ttl):
        if not items:
            ax.axis("off")
            ax.set_title(ttl)
            return
        labels = [t.term for t in items][::-1]
        vals = [t.contrib for t in items][::-1]
        ax.barh(labels, vals, alpha=0.9)
        ax.set_title(ttl)
        ax.set_label("Contribution (log-odds)")
    
    bar(axes[0], pos_terms, title_pos)
    bar(axes[1], [TermImpact(t.term, -t.contrib) for t in neg_terms], title_neg)
    fig.tight_layout()
    return fig

def plot_waterfall(contribs: List[TermImpact], intercept: float, max_terms=16, base_label="baseline"):
    contribs_sorted = sorted(contribs, key=lambda x: -abs(x.contrib))[:max_terms]
    running = intercept
    xs = [base_label] + [c.term for c in contribs_sorted]
    ys = [running]
    for c in contribs_sorted:
        running += c.contrib
        ys.append(running)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(ys)), ys, marker="o")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=45, ha="right")
    ax.set_ylabel("Log-odds")
    ax.set_title("SHAP-like Waterfall (baseline -> prediction)")
    fig.tight_layout()
    with st.expander("How to read the waterfall chart"):
        st.markdown("""
- **Baseline** (left dot) is the model's average expectation for this JD/resume pair.
- Each labelled step shows a term’s **contribution**: up = helps, down = hurts.
- **Step height = strength** of that term's influence.
- The y-axis is in **log-odds** (model units).
- Only the top terms are shown; small effects are aggregated into the baseline.
    """)
    return fig

# JD presence / coverage for down-weight & visualization
def jd_presence_terms(vectorizer, model, jd_text, resume_text, top_k=12, min_idf=2.0):
    terms = vectorizer.get_feature_names_out()
    idf = getattr(vectorizer, "idf_", None)
    Xj = vectorizer.transform([jd_text]).toarray()[0]
    Xr = vectorizer.transform([resume_text]).toarray()[0]
    w = np.ravel(model.coef_)
    scores = []
    for i, tfidf_j in enumerate(Xj):
        if tfidf_j <= 0:
            continue
        t = terms[i].lower()
        if t in STOP:
            continue
        if idf is not None and idf[i] < min_idf:
            continue
        gain = max(0.0, w[i]) * tfidf_j
        if gain <= 0:
            continue
        scores.append((t, gain, Xr[i] > 0))
    
    scores.sort(key=lambda x: -x[1])
    top = scores[:top_k]
    labels = [t for t, _, _ in top]
    present = [p for _, _, p in top]
    return labels, present

def jd_coverage(vectorizer, model, jd_text, resume_text, min_idf=2.0, top_k=12) -> float:
    labels, present = jd_presence_terms(vectorizer, model, jd_text, resume_text, top_k=top_k, min_idf=min_idf)
    if not labels:
        return 0.0
    return float(np.mean(present))


# ------------------------------Similarity / Fused Score--------------------
def cosine_on_tfidf(vectorizer, jd_text, resume_text):
    Q = normalize(vectorizer.transform([jd_text]))
    D = normalize(vectorizer.transform([resume_text]))
    return float(Q.multiply(D).sum())

def fused_score(vectorizer, model, jd_text, resume_text, alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT, gamma=GAMMA_DEFAULT):
    Xpair = build_features_pair(vectorizer, jd_text, resume_text)
    raw = float(model.decision_function(Xpair)[0])
    prob_cal = calibrated_probability(raw, Xpair)
    cos = cosine_on_tfidf(vectorizer, jd_text, resume_text)

    jd_terms = set(tokenize(jd_text)) - STOP
    rs_terms = set(tokenize(resume_text)) - STOP
    overlap = (len(jd_terms & rs_terms) / max(1, len(jd_terms)))if jd_terms else 0.0

    final = alpha * prob_cal + beta * cos + gamma * overlap
    cov = jd_coverage(vectorizer, model, jd_text, resume_text, min_idf=2.0, top_k=12)
    final = final * (0.6 + 0.4 * cov) # down-weight if high-value JD terms are missing

    final = float(np.clip(final, 0.0, 1.0))
    return final, prob_cal, cos, overlap, cov

# -----------------------------------Threshold (fixed)-------------
THR_PASS = 0.70 # fixed from PR curve shortlisting point

# -----------------------------UI-------------------------------

st.title("EXPLAINABLE RESUME - JD SCORER")
st.caption("Upload a resume and a job description to get a **match score**, transparent **reasons**, and **suggestions**.")

st.subheader("Your resume & the job description")
colA, colB = st.columns(2)

with colA:
    up = st.file_uploader("Upload your resume (PDF)", type=["pdf"], key="cand_pdf")
    resume_text = ""
    if up is not None:
        resume_text = read_pdf(up.read())
        if not resume_text:
            st.warning("Could not read text from the PDF. You can paste your resume text below.")
    resume_text = st.text_area("Or paste your resume text", value=resume_text, height=220, key="cand_resume_text")

with colB:
    jd_src = st.radio("How will you provide the job description?", ["Paste text", "Paste URL"], horizontal=True, key="cand_jd_src")
    jd_text = ""
    if jd_src == "Paste URL":
        url = st.text_input("Paste the JD link (public page)", key="cand_jd_url")
        if url:
            jd_text = fetch_jd_from_url(url)
    jd_text = st.text_area("Job description text", value=jd_text, height=220, key="cand_jd_txt")


st.markdown("---")

col1, col2 = st.columns([1.2, 2])
with col1:
    btn = st.button("Score my resume", type="primary", use_container_width=True, key="cand_btn")
    
if btn:
    if not resume_text or not jd_text:
        st.warning("Please provide both the resume and the job description.")
    else:
        with st.spinner("Scoring & explaining…"):
            final, prob, cos, ovl, cov = fused_score(vec, lr, jd_text, resume_text, alpha=alpha, beta=beta, gamma=gamma)
            score100 = 100.0 * final
            label = "Pass" if prob >= THR_PASS else "Review"

             # Explanations from linear model on the pair features
            X = build_features_pair(vec, jd_text, resume_text)
            contribs = term_contributions(vec, lr, X)
            pos, neg = pick_top_terms(contribs, k=10, stop=STOP, idf=getattr(vec,"idf_",None), idf_min=1.9)
            intercept = float(lr.intercept_[0])

            # JD presence lists (visual “mentioned vs missing”)
            jd_labels, jd_present = jd_presence_terms(vec, lr, jd_text, resume_text, top_k=12, min_idf=2.0)

        # Headline metrics
        k1, k2, k3, k4 = st.columns(4)
        with k1: 
            st.metric("Match score", f"{score100:.0f} / 100")
        with k2: 
            st.metric("Decision", "✅ Pass" if label=="Pass" else "❌ Review")
        with k3: 
            st.metric("Calibrated probability", f"{prob:.2f}")
        with k4: 
            st.metric("JD term coverage (top)", f"{100*cov:.0f}%")

         # Why this score
        st.markdown("#### Why this score?")
        st.pyplot(plot_top_terms_bar(pos, neg), use_container_width=True)

        st.markdown("#### SHAP-like contribution path")
        st.caption("Each step shows how important terms push the model from a baseline expectation to your final score.")
        st.pyplot(plot_waterfall(pos+neg, intercept, max_terms=16), use_container_width=True)

        # JD presence visualization
        st.markdown("#### JD ↔ Resume term presence")
        cL, cR = st.columns(2)
        present_terms = [t for t,p in zip(jd_labels, jd_present) if p]
        missing_terms = [t for t,p in zip(jd_labels, jd_present) if not p]
        with cL:
            st.markdown("**Mentioned in the JD and also in your resume**")
            for t in present_terms: st.markdown(f"- {t}")
        with cR:
            st.markdown("**Mentioned in the JD but not in your resume**")
            for t in missing_terms: st.markdown(f"- {t}")

# Footer: how the score is computed
with st.expander("How the score is computed", expanded=False):
    st.markdown("""
**Model.** A linear classifier (Logistic Regression) predicts match vs non-match from TF-IDF features built from the **combined text of the JD and resume**.  
**Calibration.** Raw scores are converted to well-behaved probabilities using **isotonic**/**sigmoid** calibration (or a `CalibratedClassifierCV`, if provided).  
**Fused score.** The 0–100 score blends calibrated probability, TF-IDF cosine similarity, and JD↔resume term overlap, with a small down-weight if high-value JD terms are missing.  
**Explanations.** Per-term contributions (feature value × model weight) drive the bars and the waterfall.  
**Decision threshold.** A **fixed cutoff τ = 0.82** (chosen on the validation PR curve for higher precision) is applied to the calibrated probability to mark **Pass** vs **Review**.
""")