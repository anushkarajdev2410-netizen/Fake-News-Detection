"""
Fake News Detection - Streamlit Web Application (Phase 13)

Loads the best model saved in Phase 12 (models/best_model.pkl), along with
whichever feature-extraction artifact it depends on (TF-IDF vectorizer or
Word2Vec model), and serves an interactive UI: paste article text, get a
Fake/Real prediction with a confidence score.

Run with:  streamlit run app.py
(Works regardless of which folder you run that command from -- see BASE_DIR below.)
"""

import json
import pickle
import re
import string
import html as html_lib
from pathlib import Path

import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --------------------------------------------------------------------------
# Path setup -- THIS IS THE FIX.
#
# Streamlit resolves relative paths like "models/best_model.pkl" against the
# current working directory of the process that launched it, which changes
# depending on whether you run `streamlit run app.py` from inside webapp/,
# `streamlit run webapp/app.py` from the project root, or launch it from an
# IDE "Run" button. That's what caused "Model files not found" even though
# the files genuinely exist on disk.
#
# Fixing it properly: resolve every path relative to THIS FILE's own location
# (__file__), which never changes no matter where the command is run from.
# webapp/app.py -> BASE_DIR = webapp/ -> models/ is BASE_DIR.parent / "models".
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
METADATA_PATH = MODELS_DIR / "best_model_metadata.json"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
WORD2VEC_PATH = MODELS_DIR / "word2vec.model"


# --------------------------------------------------------------------------
# Page configuration and styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="\U0001F4F0",
    layout="centered",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main {
        background-color: #F7F9FC;
    }
    .app-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1F3864;
        margin-bottom: 0;
    }
    .app-subtitle {
        font-size: 1.05rem;
        color: #5A6472;
        margin-top: 0.2rem;
        margin-bottom: 1.6rem;
    }
    .result-card {
        padding: 1.4rem 1.6rem;
        border-radius: 14px;
        margin-top: 1.2rem;
        margin-bottom: 1rem;
    }
    .result-real {
        background-color: #E5F6EF;
        border: 1.5px solid #1D9E75;
    }
    .result-fake {
        background-color: #FCEAE4;
        border: 1.5px solid #D85A30;
    }
    .result-label {
        font-size: 1.6rem;
        font-weight: 800;
    }
    .result-label-real { color: #1D9E75; }
    .result-label-fake { color: #D85A30; }
    .confidence-text {
        font-size: 0.95rem;
        color: #444;
        margin-top: 0.3rem;
    }
    .stButton>button {
        background-color: #1F3864;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.6rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2E5395;
        color: white;
    }
    .path-debug {
        font-size: 0.75rem;
        color: #999;
        font-family: monospace;
    }
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Text preprocessing -- must exactly match the pipeline used in Phase 6,
# or the model will receive differently-shaped/distributed input than it
# was trained on and predictions will silently degrade.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_nltk_resources():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    return set(stopwords.words("english")), WordNetLemmatizer()


STOPWORDS, LEMMATIZER = load_nltk_resources()

DATELINE_PATTERN = re.compile(r"^[A-Z][A-Za-z.,\s]*\((Reuters|AP|AFP)\)\s*-\s*")
HTML_TAG_PATTERN = re.compile(r"<.*?>")
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
NUMBER_PATTERN = re.compile(r"\d+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "]+",
    flags=re.UNICODE,
)
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def clean_text_pipeline(text: str):
    text = DATELINE_PATTERN.sub("", str(text))
    text = text.lower()
    text = html_lib.unescape(text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = text.translate(PUNCT_TABLE)
    text = NUMBER_PATTERN.sub(" ", text)
    text = EMOJI_PATTERN.sub(" ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return tokens


# --------------------------------------------------------------------------
# Load the trained model and matching feature-extraction artifact.
# Cached so the (relatively expensive) model/vectorizer load only happens
# once per server session, not on every button click.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    missing = [p for p in [BEST_MODEL_PATH, METADATA_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing: " + ", ".join(str(p) for p in missing)
        )

    with open(BEST_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    if metadata["feature_type"] == "word2vec":
        if not WORD2VEC_PATH.exists():
            raise FileNotFoundError(f"Missing: {WORD2VEC_PATH}")
        from gensim.models import Word2Vec
        feature_extractor = Word2Vec.load(str(WORD2VEC_PATH))
    else:
        if not TFIDF_PATH.exists():
            raise FileNotFoundError(f"Missing: {TFIDF_PATH}")
        with open(TFIDF_PATH, "rb") as f:
            feature_extractor = pickle.load(f)

    return model, metadata, feature_extractor


def vectorize(tokens, metadata, feature_extractor):
    """Turn cleaned tokens into the exact feature representation the model expects."""
    if metadata["feature_type"] == "word2vec":
        vectors = [feature_extractor.wv[t] for t in tokens if t in feature_extractor.wv]
        if len(vectors) == 0:
            return np.zeros((1, feature_extractor.vector_size))
        return np.mean(vectors, axis=0).reshape(1, -1)
    else:
        clean_string = " ".join(tokens)
        return feature_extractor.transform([clean_string])


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About this app")
    st.write(
        "This tool screens news article text for linguistic patterns "
        "associated with fabricated or unreliable reporting, using a "
        "classical machine learning model trained on the ISOT Fake and "
        "Real News dataset."
    )
    st.markdown("---")
    st.markdown(
        "**Important:** this is a content-based screening signal, not a "
        "fact-checking authority. It does not verify claims against real-world "
        "facts \u2014 it recognizes writing-style patterns. Always cross-check "
        "important claims with a trusted, independent source."
    )
    st.markdown("---")
    try:
        _, _meta, _ = load_artifacts()
        st.caption(f"Model: {_meta['model_name']}")
        st.caption(f"Test F1-score: {_meta['metrics'].get('F1', 0):.3f}")
    except Exception:
        st.caption("Model metadata unavailable.")

    with st.expander("Debug: file paths"):
        st.markdown(f'<p class="path-debug">App file: {Path(__file__).resolve()}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="path-debug">Models dir: {MODELS_DIR}</p>', unsafe_allow_html=True)
        for p in [BEST_MODEL_PATH, METADATA_PATH, TFIDF_PATH, WORD2VEC_PATH]:
            status = "\u2705 found" if p.exists() else "\u274c missing"
            st.markdown(f'<p class="path-debug">{status}: {p.name}</p>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------
st.markdown('<p class="app-title">\U0001F4F0 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Paste a news article below to screen it for '
    'patterns typically associated with fake news.</p>',
    unsafe_allow_html=True,
)

article_text = st.text_area(
    "Article text",
    height=260,
    placeholder="Paste the full article text here (the more text, the more reliable the prediction)...",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 3])
with col1:
    predict_clicked = st.button("Analyze Article", use_container_width=True)

# --------------------------------------------------------------------------
# Prediction + error handling
# --------------------------------------------------------------------------
if predict_clicked:
    stripped = article_text.strip()

    if not stripped:
        st.warning("Please paste some article text before analyzing.")
    elif len(stripped.split()) < 15:
        st.warning(
            "That text looks quite short (under 15 words). Predictions on very short "
            "snippets are much less reliable \u2014 for a meaningful result, paste a fuller "
            "excerpt or the full article body."
        )
    else:
        try:
            with st.spinner("Analyzing..."):
                model, metadata, feature_extractor = load_artifacts()
                tokens = clean_text_pipeline(stripped)

                if len(tokens) == 0:
                    st.error(
                        "After cleaning, no usable words remained in this text "
                        "(it may be entirely stopwords, numbers, or symbols). "
                        "Please try a longer, more substantive excerpt."
                    )
                else:
                    X = vectorize(tokens, metadata, feature_extractor)
                    pred = model.predict(X)[0]
                    proba = model.predict_proba(X)[0]
                    confidence = proba[pred] * 100

                    is_real = pred == 1
                    label = "REAL" if is_real else "FAKE"
                    css_class = "result-real" if is_real else "result-fake"
                    label_class = "result-label-real" if is_real else "result-label-fake"
                    icon = "\u2705" if is_real else "\u26A0\uFE0F"

                    st.markdown(
                        f"""
                        <div class="result-card {css_class}">
                            <div class="result-label {label_class}">{icon} Predicted: {label}</div>
                            <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(min(int(confidence), 100))

                    with st.expander("How was this determined?"):
                        st.write(
                            f"The model ({metadata['model_name']}) analyzed the writing "
                            f"style, vocabulary, and structure of the submitted text after "
                            f"cleaning it down to {len(tokens)} meaningful tokens, and "
                            f"compared those patterns against what it learned from "
                            f"thousands of labeled real and fake articles during training. "
                            f"It does not check any of the article's factual claims against "
                            f"outside sources."
                        )

        except FileNotFoundError as e:
            st.error(
                f"Model files not found at the expected location.\n\n"
                f"**Looked for them at:** `{MODELS_DIR}`\n\n"
                f"**Missing:** {e}\n\n"
                f"Fix: make sure `best_model.pkl`, `best_model_metadata.json`, and the matching "
                f"vectorizer or Word2Vec model are all directly inside a `models/` folder that "
                f"sits *next to* (not inside) your `webapp/` folder \u2014 i.e. `models/` and `webapp/` "
                f"should be siblings under the same project root. Expand \u201cDebug: file paths\u201d "
                f"in the sidebar to see exactly which path this app is checking and what it finds there."
            )
        except Exception as e:
            st.error(f"Something went wrong while analyzing this text: {e}")

st.markdown("---")
st.caption(
    "Built for an academic/internship project. Not intended as a substitute for "
    "professional fact-checking."
)
