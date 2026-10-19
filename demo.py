"""30-second demo: re-rank the committed HTR n-best on the held-out TEST pages and show real corrections.
Needs no download and no API key: uses data/nbest.json + data/cells_meta.json committed in this repo."""
import json, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from qc import *

meta = json.loads((ROOT / "data" / "cells_meta.json").read_text())
nb = json.loads((ROOT / "data" / "nbest.json").read_text())
runs = json.loads((ROOT / "eval_results.json").read_text())
cfg = runs[-1]["chosen_on_val"]
stats = Stats([m for m in meta if m["split"] == "train"])
val_cells = [m for m in meta if m["split"] == "val" and m["id"] in nb]
stats.fit_confusions([(nb[m["id"]][0][0], m["text"]) for m in val_cells])
test = build_context([m for m in meta if m["split"] == "test" and m["id"] in nb])
cands = [candidates([tuple(x) for x in nb[m["id"]]], stats) for m in test]
top1 = [nb[m["id"]][0][0] for m in test]
pred = rerank(test, cands, stats, cfg["weights"], cfg["margin"])
gt = [m["text"] for m in test]
n = len(gt)
helped = [i for i in range(n) if pred[i] == gt[i] != top1[i]]
harmed = [i for i in range(n) if top1[i] == gt[i] != pred[i]]
print(f"LogbookQC demo  |  {n} held-out numeric cells from {len({m['page'] for m in test})} unseen logbook pages")
print(f"weights chosen on val: {cfg}")
print(f"exact match   CTC top-1 {100*sum(a==g for a,g in zip(top1,gt))/n:.2f}%   ->   re-ranked {100*sum(a==g for a,g in zip(pred,gt))/n:.2f}%")
print(f"helped {len(helped)} cells (wrong -> right), harmed {len(harmed)} cells (right -> wrong)\n")


def show(title, idxs, k=8):
    print(title)
    print(f"  {'page':<34}{'col':<7}{'truth':>8}{'HTR top-1':>11}{'re-ranked':>11}  neighbours in same column")
    for i in idxs[:k]:
        m = test[i]
        nbr = [pred[j] for j in (m["_prev"], m["_next"]) if j is not None]
        print(f"  {m['page'][:33]:<34}{m['c'] % m['W']:<7}{gt[i]:>8}{top1[i]:>11}{pred[i]:>11}  {nbr}")
    print()


show("Corrected by the constraint layer:", helped)
show("Harmed (reported, not hidden):", harmed)
last = runs[-1]["test"]
assert abs(sum(a == g for a, g in zip(pred, gt)) / n - last["rerank_em"]) < 1e-9, "demo disagrees with eval_results.json"
print("demo result matches the last entry in eval_results.json")
