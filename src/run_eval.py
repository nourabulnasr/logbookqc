"""EVAL GATE for LogbookQC. Appends one entry per run to eval_results.json (never overwrites).

CLAIM_UNDER_TEST: Re-ranking the CTC n-best of a numeric-cell HTR model with constraints learned from
train-split ground truth (column format/range priors, temporal continuity, learned cross-field ordering,
digit-confusion candidates) raises exact-match accuracy on held-out TEST-split numeric cells by >= 1.0
percentage point over the model's top-1, while changing < 2% of the cells top-1 already got right.
"""
import sys, json, time, itertools, argparse, collections, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qc import *

CLAIM_UNDER_TEST = __doc__.split("CLAIM_UNDER_TEST:")[1].strip()
MIN_GAIN_PP, MAX_HARM = 1.0, 0.02              # pre-registered in validation.md before any run
ROOT = pathlib.Path(__file__).resolve().parent.parent


def ed(a, b):
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        q = [i]
        for j, y in enumerate(b, 1):
            q.append(min(p[j] + 1, q[j - 1] + 1, p[j - 1] + (x != y)))
        p = q
    return p[-1]


def metrics(cells, top1, pred, stats):
    gt = [m["text"] for m in cells]
    n = len(gt)
    c1 = [a == g for a, g in zip(top1, gt)]
    c2 = [a == g for a, g in zip(pred, gt)]
    harm = sum(a and not b for a, b in zip(c1, c2))
    help_ = sum(b and not a for a, b in zip(c1, c2))
    tot = sum(len(g) for g in gt)
    imp = lambda ps: sum(stats.impossible(colkey(m), p) for m, p in zip(cells, ps)) / n
    return dict(n=n, top1_em=sum(c1) / n, rerank_em=sum(c2) / n, gain_pp=100 * (sum(c2) - sum(c1)) / n,
                top1_cer=sum(ed(a, g) for a, g in zip(top1, gt)) / tot,
                rerank_cer=sum(ed(a, g) for a, g in zip(pred, gt)) / tot,
                harmed=harm, helped=help_, harm_rate=harm / max(1, sum(c1)),
                changed=sum(a != b for a, b in zip(top1, pred)),
                impossible_top1=imp(top1), impossible_rerank=imp(pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    meta = json.loads((ROOT / "data" / "cells_meta.json").read_text())
    nb = json.loads((ROOT / "data" / "nbest.json").read_text())
    train = [m for m in meta if m["split"] == "train"]
    stats = Stats(train)
    sp = {s: [m for m in meta if m["split"] == s and m["id"] in nb] for s in ("val", "test")}
    stats.fit_confusions([(nb[m["id"]][0][0], m["text"]) for m in sp["val"]])
    for s in sp:
        build_context(sp[s])
    cl = {s: [candidates([tuple(x) for x in nb[m["id"]]], stats) for m in sp[s]] for s in sp}
    top1 = {s: [tuple(nb[m["id"]][0])[0] for m in sp[s]] for s in sp}
    oracle = {s: sum(m["text"] in c for m, c in zip(sp[s], cl[s])) / len(sp[s]) for s in sp}

    # ---- choose weights on VAL only
    grid, best = [], None
    for f, r, t, p, mg in itertools.product((0, .5, 1), (0, .5, 1), (0, .5, 1), (0, 1), (0, 2, 4)):
        w = dict(fmt=f, rng=r, tmp=t, pair=p)
        pr = rerank(sp["val"], cl["val"], stats, w, mg)
        mt = metrics(sp["val"], top1["val"], pr, stats)
        grid.append(dict(w=w, margin=mg, gain_pp=mt["gain_pp"], harm_rate=mt["harm_rate"]))
        if mt["harm_rate"] < MAX_HARM and (best is None or mt["gain_pp"] > best[0]):
            best = (mt["gain_pp"], w, mg)
    if best is None:
        best = (0.0, dict(fmt=0, rng=0, tmp=0, pair=0), 4)
    _, W, MG = best
    print("val-selected weights", W, "margin", MG, "val gain_pp", round(best[0], 3))

    # ---- single evaluation on TEST
    pred = rerank(sp["test"], cl["test"], stats, W, MG)
    main_m = metrics(sp["test"], top1["test"], pred, stats)
    abl = {}
    for name in ("fmt", "rng", "tmp", "pair"):
        w2 = dict(W); w2[name] = 0
        abl[f"without_{name}"] = metrics(sp["test"], top1["test"], rerank(sp["test"], cl["test"], stats, w2, MG), stats)["gain_pp"]
        w3 = {k: (W[k] if k == name else 0) for k in W}
        abl[f"only_{name}"] = metrics(sp["test"], top1["test"], rerank(sp["test"], cl["test"], stats, w3, MG), stats)["gain_pp"]
    # per-format breakdown
    byfmt = collections.defaultdict(lambda: [0, 0, 0])
    for m, t, p in zip(sp["test"], top1["test"], pred):
        r = byfmt[fmt(m["text"])]; r[0] += 1; r[1] += t == m["text"]; r[2] += p == m["text"]
    per_field = {k: dict(n=v[0], top1_em=v[1] / v[0], rerank_em=v[2] / v[0]) for k, v in byfmt.items() if v[0] >= 30}

    gain_ok = main_m["gain_pp"] >= MIN_GAIN_PP
    harm_ok = main_m["harm_rate"] < MAX_HARM
    entry = dict(run=a.label, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), note=a.note,
                 claim_under_test=CLAIM_UNDER_TEST,
                 pre_registered=dict(min_gain_pp=MIN_GAIN_PP, max_harm_rate=MAX_HARM),
                 chosen_on_val=dict(weights=W, margin=MG), test=main_m,
                 candidate_oracle_em=oracle, ablation_gain_pp=abl, per_format=per_field,
                 constraints_learned=dict(pairs=len(stats.pairs), temporal_columns=len(stats.dscale)),
                 passes_gain=gain_ok, passes_harm=harm_ok, passes_claim=gain_ok and harm_ok)
    p = ROOT / "eval_results.json"
    hist = json.loads(p.read_text()) if p.exists() else []
    hist.append(entry)
    p.write_text(json.dumps(hist, indent=1))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"test_predictions_{a.label}.json").write_text(json.dumps(
        [dict(id=m["id"], col=colkey(m), gt=m["text"], top1=t, rerank=r) for m, t, r in zip(sp["test"], top1["test"], pred)]))
    (ROOT / "results" / f"val_grid_{a.label}.json").write_text(json.dumps(grid))
    print(json.dumps(entry, indent=1))


if __name__ == "__main__":
    main()
