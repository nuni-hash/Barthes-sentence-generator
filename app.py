import os
import tempfile
import nltk
import streamlit as st

# Use a writable temp directory for NLTK data in hosted environments
REPO_NLTK = os.path.join(os.path.abspath("."), "nltk_data")
# Prefer bundled `nltk_data` in the repo if present, otherwise use env or temp dir
NLTK_DIR = os.environ.get("NLTK_DATA_PATH") or (REPO_NLTK if os.path.isdir(REPO_NLTK) else os.path.join(tempfile.gettempdir(), "nltk_data"))
os.environ["NLTK_DATA"] = NLTK_DIR
os.makedirs(NLTK_DIR, exist_ok=True)
if NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DIR)

nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", download_dir=NLTK_DIR, quiet=True)


from new2 import MiniRecombiner

MODEL_PATH = "Barthes.txt"

st.title("Mini Recombiner")
st.caption("Click Generate to produce short recombined sentences from Barthes' essay")


n_sent = st.slider("Sentences per click", 1, 10, 5)
if "model" not in st.session_state:
    with st.spinner("Loading model (first run may take a few seconds)..."):
        st.session_state.model = MiniRecombiner(MODEL_PATH)

if st.button("Generate"):
    with st.spinner("Generating..."):
        out = []
        for _ in range(n_sent):
            s = st.session_state.model.generate_sentence()
            if s:
                out.append(s)
    if out:
        st.text_area("Output", value="\n".join(out), height=200)
    else:
        st.info("No sentence generated. Try again.")