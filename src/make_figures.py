"""Render results/eval_summary.png from eval_results.json (every run, in order). No numbers are typed by hand."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
runs = json.loads((ROOT / "eval_results.json").read_text())
ink, accent, grey = "#1d1d1f", "#b4452c", "#9a948a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw=dict(width_ratios=[1.1, 1.3, 1.2]))

# 1) exact match, top-1 vs re-ranked, every run
labels, t1, rr = [], [], []
for r in runs:
    labels.append(r["run"]); t1.append(100 * r["test"]["top1_em"]); rr.append(100 * r["test"]["rerank_em"])
x = range(len(runs))
ax[0].bar([i - .2 for i in x], t1, .4, color=grey, label="CTC top-1")
ax[0].bar([i + .2 for i in x], rr, .4, color=accent, label="constraint re-ranked")
for i in x:
    ax[0].text(i - .2, t1[i] + .3, f"{t1[i]:.1f}", ha="center", fontsize=8)
    ax[0].text(i + .2, rr[i] + .3, f"{rr[i]:.1f}", ha="center", fontsize=8)
ax[0].set_xticks(list(x)); ax[0].set_xticklabels(labels, fontsize=8)
ax[0].set_ylim(min(t1 + rr) - 4, 100); ax[0].set_ylabel("test exact match (%)")
ax[0].set_title("Exact match, held-out test cells"); ax[0].legend(frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.32))

# 2) per-constraint ablation (last run)
last = runs[-1]
ab = last["ablation_gain_pp"]
names = ["fmt", "rng", "tmp", "pair"]
yy = range(len(names))
ax[1].barh([i + .2 for i in yy], [ab[f"only_{n}"] for n in names], .4, color=accent, label="only this constraint")
ax[1].barh([i - .2 for i in yy], [ab[f"without_{n}"] for n in names], .4, color=grey, label="all but this one")
ax[1].axvline(last["test"]["gain_pp"], color=ink, lw=.8, ls="--"); ax[1].axvline(0, color=ink, lw=.5)
ax[1].set_yticks(list(yy)); ax[1].set_yticklabels(["format", "range", "temporal", "pairwise"])
ax[1].set_xlabel("gain over top-1 (pp)  [dashed = full model]"); ax[1].set_title(f"Ablation ({last['run']})")
ax[1].legend(frameon=False, fontsize=8)

# 3) harm / help in cells
t = last["test"]
ax[2].bar(["helped\n(wrong→right)", "harmed\n(right→wrong)", "changed\n(total)"],
          [t["helped"], t["harmed"], t["changed"]], color=[accent, ink, grey])
for i, v in enumerate([t["helped"], t["harmed"], t["changed"]]):
    ax[2].text(i, v + .3, str(v), ha="center")
ax[2].set_title(f"What re-ranking did ({t['n']} test cells)\nharm rate {100*t['harm_rate']:.2f}% of top-1-correct")
fig.tight_layout()
out = ROOT / "results" / "eval_summary.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=140)
print("wrote", out)
