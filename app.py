import os
import nltk
import streamlit as st

NLTK_DIR = os.path.join(os.path.abspath("."), "nltk_data")
os.environ["NLTK_DATA"] = NLTK_DIR          # ensure NLTK will look here
os.makedirs(NLTK_DIR, exist_ok=True)
if NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DIR)

# safe: download calls are optional because postBuild will handle it
nltk.download("punkt", download_dir=NLTK_DIR, quiet=True)
nltk.download("averaged_perceptron_tagger", download_dir=NLTK_DIR, quiet=True)

# Import the model after NLTK is configured
from new2 import MiniRecombiner

MODEL_PATH = "Barthes.txt"

st.title("Mini Recombiner")
st.caption("Click Generate to produce short, grammatical sentences from the corpus.")

# controls
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