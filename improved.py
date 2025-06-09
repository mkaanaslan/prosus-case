import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Set

import litellm
import numpy as np
import pandas as pd

from baseline import (
    META_DATA,
    PRED_DIR,
    RESULTS_DIR,
    TEST,
    search_index,
    get_results,
    search_index
)

from collections import defaultdict
from typing import List, Sequence

TAX_L2, TAX_L1, TAX_L0 = (defaultdict(list) for _ in range(3))

for row in META_DATA.itertuples():
    TAX_L2[row.l2].append(row[1])
    TAX_L1[row.l1].append(row[1])
    TAX_L0[row.l0].append(row[1])

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "related_item_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 5,
            "maxItems": 5,
        },
        "alt_queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 5,
            "maxItems": 5,
        },
    },
    "required": ["related_item_ids", "alt_queries"],
    "additionalProperties": False,
}


LLM_PROMPT = """You are a large-language model that specializes in product-search orchestration.  
Your job is to (1) pick *exactly five* “pivot” items from an initial retrieval list and (2) invent *exactly five* alternate search queries that could surface additional relevant products.

**Original query**
{original_query}

**Top-100 candidates**  
Each line contains the internal ID, the product name, and its taxonomy path (from leaf → root):
{topk}


### Task A – Select 5 pivot items  
Choose 5 item IDs that will work well as starting points for *taxonomy-tree expansion*.  
Guidelines for a good set:

1. **Coverage** – Aim to cover different meaningful branches of the taxonomy (different categories or different `l2` / `l1` levels) so that searching neighbours will discover diverse, yet still relevant, products.  
2. **Specificity** – Favour items that sit at a reasonably specific node (neither too generic nor ultra-niche).  
3. **Relevance** – All chosen items must be clearly relevant to the original query.  
4. **No duplicates** – Obviously pick distinct IDs.

### Task B – Generate 5 alternate queries  
Write 5 concise natural-language queries that capture additional facets of the user’s intent. 
*Think synonyms, regional wording, ingredient focus, brand variants, usage occasions, etc.*  
These queries will be re-run through the retrieval engine, so they should be **stand-alone** (do *not* reference the list above) and **varied** (avoid trivial rewrites).

---

### Output specification  
Return **one** JSON object that conforms exactly to the schema below – no extra keys, no commentary, no markdown.

"""

LLM = "gpt-4.1-mini"

W_INIT = 2.0       # original query
W_ALTQ = 0.3       # each alt query
W_TAX  = 0.5       # taxonomy expansion
K_RRF  = 15        # denominator constant
TOP_K  = 300       # how many you keep for eval / re-rank
INIT_K = 100

def initial_retrieval(queries: List[str], k: int = INIT_K) -> List[List[int]]:
    results = search_index(queries, index_name="nlp", top_k=k)
    ids_initial = [r["_id"] for r in results]
    prompts_initial = []
    for batch in ids_initial:
        lines = []
        for id in batch:
            rec = META_DATA.loc[META_DATA._id == id].iloc[0]
            prompt_line = f"ID: {rec['_id']} | {rec['name']} -> {rec['category']} -> {rec['l0']} -> {rec['l1']} -> {rec['l2']}"
            lines.append(prompt_line)
        prompts_initial.append("\n".join(lines))
    return results, prompts_initial


def llm_expansion(queries: List[str], k: int = INIT_K) -> List[List[int]]:
    initial_results, initial_prompts = initial_retrieval(queries)
    prompts = [LLM_PROMPT.format(original_query=original_query, topk=topk) + str(_JSON_SCHEMA) for original_query, topk in zip(queries, initial_prompts)]
    messages = [[{"role": "user", "content": user_msg}] for user_msg in prompts]
    resp = litellm.batch_completion(
        model=LLM,
        messages=messages,
        response_format = { "type": "json_schema", "json_schema": { "name": "ExpandResponse", "strict": True, "schema": _JSON_SCHEMA } },
        api_base=os.environ["LITELLM_API_BASE"],
        api_key=os.environ["LITELLM_API_KEY"],
        seed=0,
        temperature=0.2,
    )
    outputs = [json.loads(r["choices"][0]["message"]["content"]) for r in resp]
    return initial_results, outputs


def improved_search(queries: List[str]):
    initial_results, outputs = llm_expansion(queries)
    all_results = []
    for initial_result, out in zip(initial_results, outputs):
        new_retrieval = search_index(out["alt_queries"], index_name="nlp", top_k=INIT_K)
        taxonomy_results = taxonomy_expand(out['related_item_ids'])
        rank_lists = [
            initial_result["_id"],                # original query hits
            *[q["_id"] for q in new_retrieval],   # each alt‐query hit list
            taxonomy_results                      # taxonomy‐expansion list
        ]
        weights    =  [W_INIT]       + [W_ALTQ]*len(new_retrieval) + [W_TAX]
        final_ids = wrrf_fuse(rank_lists, weights, k=K_RRF, top_k=TOP_K)
        all_results.append(final_ids)
    return all_results



def taxonomy_expand(
    seed_ids: Sequence[int],
    quotas: Tuple[int, int, int] = (10, 6, 4),   # how many neighbours to keep per level (l2, l1, l0)
    target_total: int = 100,
) -> List[int]:
    seen:  Set[int] = set(seed_ids)  # avoid duplicates
    ranked: List[int] = []

    for sid in seed_ids:
        row = META_DATA.loc[META_DATA._id == sid]
        if row.empty:
            continue
        meta = row.iloc[0]

        added = 0
        for pid in TAX_L0[meta.l0]:
            if pid not in seen:
                ranked.append(pid);  seen.add(pid);  added += 1
                if added >= quotas[0] or len(ranked) >= target_total:
                    break

        added = 0
        for pid in TAX_L1[meta.l1]:
            if pid not in seen:
                ranked.append(pid);  seen.add(pid);  added += 1
                if added >= quotas[1] or len(ranked) >= target_total:
                    break

        added = 0
        for pid in TAX_L2[meta.l2]:
            if pid not in seen:
                ranked.append(pid);  seen.add(pid);  added += 1
                if added >= quotas[2] or len(ranked) >= target_total:
                    break

        if len(ranked) >= target_total:
            break

    if len(ranked) < target_total:
        for sid in seed_ids:
            meta = META_DATA.loc[META_DATA._id == sid].iloc[0]
            for pid in TAX_L0[meta.l0]:
                if pid not in seen:
                    ranked.append(pid);  seen.add(pid)
                    if len(ranked) >= target_total:
                        break
            if len(ranked) >= target_total:
                break

    return ranked[:target_total]


def wrrf_fuse(rank_lists, weights, k: int = 60, top_k: int = 100):
    scores = {}
    for lst, w in zip(rank_lists, weights):
        for r, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0) + w / (k + r + 1)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [doc_id for doc_id, _ in fused[:top_k]]


def save_and_eval(
    preds: List[List[int | str]],
    out_tag: str = "llm_fusion",
) -> Dict[str, float]:

    test_out = TEST.copy()
    test_out["preds"] = preds

    out_json = PRED_DIR / f"{out_tag}_preds.json"
    test_out.to_json(out_json, orient="records", force_ascii=False, indent=0)
    print(f"✓ Saved predictions to {out_json}")

    metrics = get_results(out_json)
    print("=== Results ===")
    for k, v in metrics.items():
        if k != "type":
            print(f"{k:<10s}: {v:.4f}")

    res_df = pd.DataFrame([metrics])
    res_df.to_csv(RESULTS_DIR / f"{out_tag}_results.csv", index=False)
    with open(RESULTS_DIR / f"{out_tag}_results.md", "w", encoding="utf8") as fp:
        fp.write(res_df.to_markdown(index=False))

    print(f"✓ Eval artifacts written to {RESULTS_DIR}/")
    return metrics



if __name__ == "__main__":
    
    queries = TEST.query_text.tolist()
    search_results = improved_search(queries)
    metrics = save_and_eval(search_results, out_tag="improved")