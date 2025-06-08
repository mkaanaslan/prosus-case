import os
import json
from pathlib import Path
import pickle
from rank_bm25 import BM25Okapi
import faiss
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import litellm
from typing import Union, List, Dict, Any

UPDATE      = True
CSV_PATH    = Path("data/5k_items_curated.csv")
MODEL_NAME  = "text-embedding-3-large"
VEC_DIR     = Path("embeddings");  VEC_DIR.mkdir(exist_ok=True)
PRED_DIR    = Path("preds");       PRED_DIR.mkdir(exist_ok=True)
TEST_PATH   = Path("data/test.json")
RESULTS_DIR = Path("results");     RESULTS_DIR.mkdir(exist_ok=True)

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
def get_embeddings(texts, chunk_size: int = 128, verbose: bool = False):
    texts = [t if str(t).strip() else "None" for t in texts]
    all_embs = []
    for start in tqdm(range(0, len(texts), chunk_size), disable=not verbose):
        chunk = texts[start:start + chunk_size]
        resp = litellm.embedding(
            model=MODEL_NAME,
            input=chunk,
            api_base=os.environ["LITELLM_API_BASE"],
            api_key=os.environ["LITELLM_API_KEY"],
        )
        embs = np.array([d["embedding"] for d in resp["data"]], dtype=np.float32)
        all_embs.append(embs)

    embeddings = np.vstack(all_embs)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def build_index(texts, index_name, chunk_size = 128):
    index_path = VEC_DIR / f"{index_name}.faiss"
    embeddings = get_embeddings(texts, chunk_size=chunk_size, verbose=True)
    dim        = embeddings.shape[1]

    idx = faiss.IndexFlatIP(dim)
    idx.add(embeddings)
    faiss.write_index(idx, str(index_path))
    print(f"✓ Built and saved index '{index_name}' with {idx.ntotal} vectors")

def search_index(queries: Union[str, List[str]],
                 index_name: str,
                 top_k: int = 10,
                 chunk_size: int = 128,
                 verbose: bool = False
                 ) -> List[Dict[str, Any]]:
    if isinstance(queries, str):
        queries = [queries]

    q_embs = get_embeddings(queries, chunk_size=chunk_size, verbose=verbose)

    index_path = VEC_DIR / f"{index_name}.faiss"
    idx = faiss.read_index(str(index_path))

    scores, indices = idx.search(q_embs, top_k)

    results = []
    for row_scores, row_idxs in zip(scores, indices):
        results.append({
            "_id":   [IDMAP[i]   for i in row_idxs],
            "score": [float(s)    for s in row_scores],
        })

    return results


# ------------------ weighted-RRF helpers ------------------ #
def wrrf_fuse(rank_lists, weights, k: int = 60, top_k: int = 100):
    scores = {}
    for lst, w in zip(rank_lists, weights):
        for r, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0) + w / (k + r + 1)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [doc_id for doc_id, _ in fused[:top_k]]


def get_wrrf_predictions(idx_a: str, idx_b: str,
                         w_a: float, w_b: float,
                         k: int = 60, top_k: int = 100):
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
def get_predictions(index_name: str,
                    top_k: int = 10,
                    chunk_size: int = 128,
                    verbose: bool = False):
    with open(TEST_PATH, "r", encoding="utf8") as f:
        test = json.load(f)

    if index_name.startswith("bm25"):
        for i, item in enumerate(test):
            res = search_bm25_index(item["query"], index_name, top_k)
            test[i]["preds"] = res["_id"]
    else:
        queries = [item["query"] for item in test]
        batch_results = search_index(
            queries,
            index_name=index_name,
            top_k=top_k,
            chunk_size=chunk_size,
            verbose=verbose
        )
        for item, res in zip(test, batch_results):
            item["preds"] = res["_id"]

    out_path = PRED_DIR / f"{index_name}_preds.json"
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(test, f, ensure_ascii=False, indent=0)

    print(f"✓ Saved predictions for '{index_name}' to {out_path}")


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

        pd.DataFrame(all_results).to_csv(RESULTS_DIR / "baseline_results.csv", index=False)
        with open(RESULTS_DIR / "baseline_results.md", "w") as f:
            f.write(pd.DataFrame(all_results).to_markdown())
