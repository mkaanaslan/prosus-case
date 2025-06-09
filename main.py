import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# Internal imports – assumes baseline.py and improved.py are in the same directory
# and that all required vector stores / indices have already been built.
# --------------------------------------------------------------------------------------
#from improved import improved_search  # noqa: E402  (delayed import for Streamlit)
from baseline import search_index

########################################################################################
# Utility helpers                                                                      #
########################################################################################

def load_catalog(csv_path: Path = Path("data/5k_items_curated.csv")) -> Dict[str, Dict[str, str]]:
    """Load product metadata and prepare fast lookup dictionaries.

    Returns
    -------
    dicts
        - names:  _id → product name
        - images: _id → image_str (may be empty)
    """
    df = pd.read_csv(csv_path)

    # Parse *once* and stash JSON to avoid repeated `json.loads` during rendering
    df["_meta"] = df["itemMetadata"].apply(json.loads)
    names  = df.set_index("_id")["_meta"].apply(lambda m: m.get("name", "")).to_dict()

    def _extract_image(m: dict) -> str | None:
        # According to the spec, the frontend should hit
        #   https://static.ifood-static.com.br/image/upload/t_low/pratos/<image_str>
        # for each item.  The CSV generally stores an *array* of images.  When that is
        # missing or empty we gracefully return `None`.
        imgs: list = m.get("images") or []
        if imgs:
            # Each entry can be either a raw string or a nested dict.  Handle both.
            first = imgs[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                # Most common key patterns observed:
                #   { "id": "abcdef" } or { "imageId": "abcdef" }
                return first.get("id") or first.get("imageId") or first.get("key")
        return None

    images = df.set_index("_id")["_meta"].apply(_extract_image).to_dict()
    return names, images


# Cache catalog in memory so Streamlit refreshes remain snappy
@st.cache_data(show_spinner=False)
def get_catalog():
    return load_catalog()


# --------------------------------------------------------------------------------------
# Streamlit UI                                                                         #
# --------------------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="iFood Product Search", layout="wide")
    st.title("🍕🔍 iFood Product Search Demo")

    # ------------ Sidebar options ------------ #
    st.sidebar.header("Settings")
    top_k = st.sidebar.slider("Number of results", min_value=10, max_value=100, value=40, step=10)

    # ------------ Search bar ------------ #
    query = st.text_input("Enter your search query", placeholder="e.g. chocolate cake 200g")

    # Load catalog metadata once
    names, images = get_catalog()

    # ------------ Result panel ------------ #
    if query:
        with st.spinner("Generating recommendations …"):
            try:
                # improved_search expects a list of queries and returns a list of id lists
                #preds: List[str] = improved_search([query])[0][:top_k]
                preds: List[str] = search_index(query, index_name="nlp", top_k=top_k)[0]['_id']
            except Exception as exc:
                st.error(f"Search failed: {exc}")
                st.stop()

        if not preds:
            st.warning("No results found. Try a different query ☝️")
            st.stop()

        st.markdown(f"### Top {len(preds)} recommendations for **'{query}'**")
        n_cols = 4  # number of tiles per row
        cols = st.columns(n_cols, gap="large")

        for idx, pid in enumerate(preds):
            col = cols[idx % n_cols]
            with col:
                img_key = images.get(pid)
                if img_key:
                    img_url = f"https://static.ifood-static.com.br/image/upload/t_low/pratos/{img_key}"
                    st.image(img_url, use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/256?text=No+Image", use_container_width=True)
                st.caption(names.get(pid, "<Unnamed Product>"))


if __name__ == "__main__":
    main()
