"""Synthetic ground-truth checks: the re-ranker must fix an injected impossible/discontinuous misread and
must NOT touch a plausible reading when the model is confident."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import numpy as np
from qc import Stats, build_context, rerank, candidates, colkey


def make_page(rng, page, n=12):
    cells = []
    base = 30.0
    for r in range(n):
        base += rng.normal(0, 0.03)
        cells.append(dict(page=page, table="table01", fam="vol003", W=16, c=16 * r + 5, text=f"{base:.2f}", id=f"{page}_{r}"))
    return cells


def setup():
    rng = np.random.default_rng(0)
    train = sum((make_page(rng, f"p{i}") for i in range(30)), [])
    return Stats(train), rng


def test_fixes_impossible_digit_with_neighbour_context():
    stats, rng = setup()
    page = make_page(rng, "t0")
    build_context(page)
    k = 5
    truth = page[k]["text"]
    wrong = "3" + chr(ord(truth[1]) + 0) + truth[2:]            # keep valid
    wrong = "8" + truth[1:]                                     # 80.xx: impossible for a barometer column
    cl = []
    for i, m in enumerate(page):
        if i == k:
            cl.append({wrong: -0.2, truth: -1.5})              # model slightly prefers the wrong read
        else:
            cl.append({m["text"]: -0.05})
    pred = rerank(page, cl, stats, dict(fmt=1, rng=1, tmp=1, pair=0), margin=0)
    assert pred[k] == truth


def test_margin_protects_confident_reading():
    stats, rng = setup()
    page = make_page(rng, "t1")
    build_context(page)
    cl = [{m["text"]: -0.01, "99.99": -6.0} for m in page]
    pred = rerank(page, cl, stats, dict(fmt=1, rng=1, tmp=1, pair=0), margin=2)
    assert pred == [m["text"] for m in page]


def test_confusion_candidates_added():
    stats, _ = setup()
    stats.fit_confusions([("3", "8")] * 20 + [("3", "3")] * 5)
    c = candidates([("30.31", -0.3)], stats)
    assert "80.31" in c or "30.81" in c
