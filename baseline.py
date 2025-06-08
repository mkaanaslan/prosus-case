import os
import json
from pathlib import Path
import pickle
from rank_bm25 import BM25Okapi
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

UPDATE      = True
CSV_PATH    = Path("data/5k_items_curated.csv")
MODEL_NAME  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VEC_DIR     = Path("embeddings");  VEC_DIR.mkdir(exist_ok=True)
PRED_DIR    = Path("preds");       PRED_DIR.mkdir(exist_ok=True)
TEST_PATH   = Path("data/test.json")
RESULTS_DIR = Path("results");     RESULTS_DIR.mkdir(exist_ok=True)
SEED        = 42

MODEL = SentenceTransformer(MODEL_NAME)

# -------------------------- data loading --------------------------- #

def get_metadata(row):
    meta     = json.loads(row.itemMetadata)
    profile  = json.loads(row.itemProfile)
    return {
        "_id":         row._id,
        "name":        meta["name"],
        "category":    meta["category_name"],
        "l0":          meta["taxonomy"]["l0"],
        "l1":          meta["taxonomy"]["l1"],
        "l2":          meta["taxonomy"]["l2"],
        "co_purchase": [i["item_name"] for i in profile["metrics"]["coPurchaseItems"] if "item_name" in i],
        "search":      [i["term"] for i in profile["search"] if "term" in i],
        "ordering_rate": {**profile['metrics']['orderingRate']['day'], **profile['metrics']['orderingRate']['shift']},
    }

DATA       = pd.read_csv(CSV_PATH)
META_DATA  = DATA.apply(get_metadata, axis=1, result_type="expand")
IDMAP        = DATA["_id"].tolist()


def describe_ordering_pattern(probs: dict[str, float],
                              day_diff_thresh: float = 0.10,
                              uniform_thresh: float = 0.05) -> str:
    weekday = probs.get("weekday")
    weekend = probs.get("weekend")

    day_phrase = None
    day_uniform = False
    if weekday is not None and weekend is not None:
        gap = abs(weekday - weekend)
        if gap < day_diff_thresh:
            day_phrase = "tanto em dias úteis quanto em finais de semana"
            day_uniform = True
        else:
            day_phrase = "principalmente em dias úteis" if weekday > weekend else "principalmente nos finais de semana"
    elif weekday is not None:
        day_phrase = "principalmente em dias úteis"
    elif weekend is not None:
        day_phrase = "principalmente nos finais de semana"

    shift_keys = [k for k in probs if k not in ("weekday", "weekend")]
    shifts = [(k, probs[k]) for k in shift_keys]

    shift_phrase = None
    shift_uniform = False
    if shifts:
        n = len(shifts)
        uniform = 1 / n
        if all(abs(p - uniform) < uniform_thresh for _, p in shifts):
            shift_uniform = True
        else:
            shifts.sort(key=lambda kv: kv[1], reverse=True)
            top, top_p = shifts[0]
            runners_up = [k for k, v in shifts[1:] if top_p - v < uniform_thresh]

            nice = lambda s: s.replace("_", " ").lower()
            if runners_up:
                names = [nice(top)] + [nice(r) for r in runners_up]
                if len(names) == 2:
                    shift_phrase = f"{names[0]} e {names[1]}"
                else:
                    shift_phrase = ", ".join(names[:-1]) + f", e {names[-1]}"
            else:
                shift_phrase = nice(top)

    if (shift_uniform or not shift_phrase) and (day_uniform or not day_phrase):
        return "Este item pode ser pedido a qualquer momento."

    sentence = "Este item é geralmente preferido"
    if shift_phrase:
        sentence += f" em {shift_phrase}"
    if day_phrase:
        sentence += (", " if shift_phrase else " ") + day_phrase + "."
    else:
        sentence += "."

    return sentence


# -------------------------- BM25 helpers --------------------------- #
def build_bm25_index(texts, index_name: str):
    tokenised = [t.lower().split() for t in texts]
    bm25      = BM25Okapi(tokenised)
    with open(VEC_DIR / f"{index_name}.pkl", "wb") as fp:
        pickle.dump(bm25, fp)
    print(f"✓ Built and saved BM25 index '{index_name}' with {len(texts)} docs")


def search_bm25_index(query: str, index_name: str, top_k: int = 10):
    with open(VEC_DIR / f"{index_name}.pkl", "rb") as fp:
        bm25 = pickle.load(fp)

    scores = bm25.get_scores(query.lower().split())
    best   = np.argsort(scores)[::-1][:top_k]

    return {"_id": [IDMAP[i] for i in best],
            "score": [float(scores[i]) for i in best]}


# ------------------------- FAISS helpers --------------------------- #
def get_embeddings(texts, verbose=False):
    return MODEL.encode(texts,
                        batch_size=128,
                        show_progress_bar=verbose,
                        normalize_embeddings=True)


def build_index(texts, index_name):
    index_path = VEC_DIR / f"{index_name}.faiss"
    embeddings = get_embeddings(texts, verbose=True)
    dim        = embeddings.shape[1]

    idx = faiss.IndexFlatIP(dim)
    idx.add(embeddings)
    faiss.write_index(idx, str(index_path))
    print(f"✓ Built and saved index '{index_name}' with {idx.ntotal} vectors")


def search_index(query: str, index_name: str, top_k: int = 10):
    qvec       = get_embeddings([query])
    index_path = VEC_DIR / f"{index_name}.faiss"
    idx        = faiss.read_index(str(index_path))

    scores, indices = idx.search(qvec, top_k)
    return {"_id":   [IDMAP[i] for i in indices[0]],
            "score": [float(s)  for s in scores[0]]}

# ------------------ weighted-RRF helpers ------------------ #
def wrrf_fuse(rank_lists, weights, k: int = 60, top_k: int = 100):
    """
    Weighted Reciprocal Rank Fusion: score = w / (k + rank).
    `weights` must be the same length as `rank_lists`.
    """
    scores = {}
    for lst, w in zip(rank_lists, weights):
        for r, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0) + w / (k + r + 1)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [doc_id for doc_id, _ in fused[:top_k]]


def get_wrrf_predictions(idx_a: str, idx_b: str,
                         w_a: float, w_b: float,
                         k: int = 60, top_k: int = 100):
    """
    Build & save a weighted-RRF run; returns its run-name.
    """
    pa, pb = PRED_DIR / f"{idx_a}_preds.json", PRED_DIR / f"{idx_b}_preds.json"
    with open(pa, "r", encoding="utf8") as fa, open(pb, "r", encoding="utf8") as fb:
        ta, tb = json.load(fa), json.load(fb)

    fused = []
    for qa, qb in zip(ta, tb):
        fused_preds = wrrf_fuse([qa["preds"], qb["preds"]],
                                weights=[w_a, w_b],
                                k=k, top_k=top_k)
        fused.append({"_id": qa["_id"], "query": qa["query"], "preds": fused_preds})

    run_name = f"wrrf_k{k}_w{int(w_b*100)}_{idx_a}_{idx_b}"
    with open(PRED_DIR / f"{run_name}_preds.json", "w", encoding="utf8") as f:
        json.dump(fused, f, ensure_ascii=False, indent=0)
    return run_name



# ---------------------- prediction + metrics ----------------------- #
def get_predictions(index_name: str, top_k: int = 10):
    with open(TEST_PATH, "r", encoding="utf8") as f:
        test = json.load(f)

    for i, item in enumerate(test):
        if index_name.startswith("bm25"):
            res = search_bm25_index(item["query"], index_name, top_k)
        else:
            res = search_index(item["query"], index_name, top_k)
        test[i]["preds"] = res["_id"]

    with open(PRED_DIR / f"{index_name}_preds.json", "w", encoding="utf8") as f:
        json.dump(test, f, ensure_ascii=False, indent=0)


def get_results(preds_path: str):
    with open(preds_path, "r", encoding="utf8") as f:
        test = json.load(f)

    recalls = {1: 0, 3: 0, 5: 0, 10: 0, 20:0, 30:0, 50:0, 100:0}
    for item in test:
        for k in recalls:
            if item["_id"] in item["preds"][:k]:
                recalls[k] += 1

    total = len(test)
    res   = {f"Recall@{k}": recalls[k] / total for k in recalls}
    res['type'] = preds_path.stem

    return res


if __name__ == "__main__":

    if UPDATE:

        with open(VEC_DIR / "idmap.json", "w", encoding="utf8") as fp:
            json.dump(IDMAP, fp, ensure_ascii=False, indent=0)

        # ---------------------- text field variants ------------------------ #
        name_texts = META_DATA["name"].tolist()

        taxonomy_template = "{cat} - {l2} - {l1} - {l0}"
        taxonomy_texts = [
            taxonomy_template.format(
                cat=x.category,
                l0=x.l0.replace("_", " "),
                l1=x.l1.replace("_", " "),
                l2=x.l2.replace("_", " "),
            )
            for x in META_DATA.itertuples()
        ]

        name_tax_texts = [f"{n}. {t}" for n, t in zip(name_texts, taxonomy_texts)]

        full_texts = []
        for x in META_DATA.itertuples():
            co_p   = ", ".join(x.co_purchase) if x.co_purchase else "nada"
            search = ", ".join(x.search)      if x.search      else "nenhum"
            full_texts.append(
                f"{x.name}. Categoria: {x.category}. "
                f"Taxonomia: {x.l2.replace('_',' ')} > {x.l1.replace('_',' ')} > {x.l0.replace('_',' ')}. "
                f"Também compram: {co_p}. Termos de busca: {search}."
            )

        nlp_texts = []
        for x in META_DATA.itertuples():
            co_p   = ", ".join(x.co_purchase) if x.co_purchase else "nenhum"
            search = ", ".join(x.search)      if x.search      else "nenhum"
            ordering = describe_ordering_pattern(x.ordering_rate)

            nlp_texts.append(
                f"{x.name} é um produto da categoria {x.category}. "
                f"Taxonomia: {x.l2.replace('_',' ')} > {x.l1.replace('_',' ')} > {x.l0.replace('_',' ')}. "
                f"Clientes que compram este item também compram: {co_p}. "
                f"Termos de busca frequentes: {search}. "
                f"{ordering}"
            )

        # ---------------------------- build -------------------------------- #
        for idx, texts in [
            ("name",               name_texts),
            ("taxonomy",           taxonomy_texts),
            ("co_purchase",        META_DATA["co_purchase"].apply(lambda z: " ".join(z)).tolist()),
            ("search",             META_DATA["search"].apply(lambda z: " ".join(z)).tolist()),
            ("full",               full_texts),
            ("nlp",          nlp_texts),
        ]:
            build_index(texts, idx)

        build_bm25_index(name_texts,        "bm25_name")
        build_bm25_index(name_tax_texts,    "bm25_name_taxonomy")
        build_bm25_index(full_texts,        "bm25_full")
        build_bm25_index(nlp_texts,         "bm25_nlp") 

        # ------------------------ evaluation loop -------------------------- #
        all_indices = [
            "name", "taxonomy", "co_purchase", "search", "full",
            "bm25_name", "bm25_name_taxonomy", "bm25_full", "nlp", "bm25_nlp" 
        ]

        all_results = []
        for idx in all_indices:
            get_predictions(index_name=idx, top_k=100)
            cur_res = get_results(preds_path=PRED_DIR / f"{idx}_preds.json")
            all_results.append(cur_res)

        grid_results = []
        anchor_idx   = "bm25_name"
        anchor_r30  = pd.DataFrame(all_results).loc[pd.DataFrame(all_results).type == f"{anchor_idx}_preds",
                                        "Recall@30"].iloc[0]

        k_grid = [20, 40, 60, 80]
        w_grid = [0.25, 0.5, 0.75, 1.0]
        best_pair, best_r30 = None, anchor_r30

        for k_val in k_grid:
            for w_emb in w_grid:
                run_name = get_wrrf_predictions(anchor_idx, "nlp",
                                                w_a=1.0, w_b=w_emb,
                                                k=k_val, top_k=100)
                res = get_results(PRED_DIR / f"{run_name}_preds.json")
                grid_results.append(res)
                os.remove(PRED_DIR / f"{run_name}_preds.json")

                if res["Recall@30"] > best_r30:
                    best_pair, best_r30, best_res = run_name, res["Recall@30"], res

        with open(RESULTS_DIR / "baseline_grid_wrrf.md", "w") as f:
            f.write(pd.DataFrame(grid_results).to_markdown())
        print(f"Best weighted-RRF run: {best_pair}  (Recall@30 = {best_r30:.3f})")
        all_results.append(best_res)

        all_results = pd.DataFrame(all_results)
        all_results.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)
        with open(RESULTS_DIR / "baseline_results.md", "w") as f:
            f.write(all_results.to_markdown())
