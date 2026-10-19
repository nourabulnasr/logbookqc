"""LogbookQC: physics/format-constrained n-best re-ranker for handwritten numeric log cells.

All constraints are LEARNED from train-split ground truth (no modern units are hard-coded):
  fmt   per-column digit-format prior       (e.g. a barometer column is always dd.dd)
  rng   per-column value-range prior        (histogram of train values)
  tmp   temporal continuity                 (Laplace model of change between consecutive same-column cells)
  pair  cross-field ordering                (column pairs where A>=B holds >=97% of the time in train)
plus digit-confusion substitution candidates learned from the HTR model's own validation errors.
"""
import math, re, collections, itertools
import numpy as np

NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
BAD = -15.0                                    # log-prior for strings that are not numbers


def fmt(s):
    return re.sub(r"\d", "d", s)


def val(s):
    return float(s) if NUM_RE.match(s) else None


def colkey(m):
    return f"{m['fam']}|{m['W']}|{m['c'] % m['W']}"


class Stats:
    def __init__(self, train):
        self.fmt = collections.defaultdict(collections.Counter)
        self.fmt_fam = collections.defaultdict(collections.Counter)
        vals = collections.defaultdict(list)
        pages = collections.defaultdict(list)
        for m in train:
            k = colkey(m)
            self.fmt[k][fmt(m["text"])] += 1
            self.fmt_fam[m["fam"]][fmt(m["text"])] += 1
            vals[k].append(float(m["text"]))
            pages[(m["page"], m["table"])].append(m)
        self.vals = {k: np.sort(np.array(v)) for k, v in vals.items()}
        self.lo = {k: v[0] for k, v in self.vals.items()}
        self.hi = {k: v[-1] for k, v in self.vals.items()}
        dl = collections.defaultdict(list)
        pair = collections.defaultdict(lambda: [0, 0, 0])   # n, a>=b, a<=b
        for ms in pages.values():
            ms = sorted(ms, key=lambda m: m["c"])
            bycol = collections.defaultdict(list)
            for m in ms:
                bycol[colkey(m)].append(float(m["text"]))
            for k, v in bycol.items():
                dl[k] += [abs(b - a) for a, b in zip(v, v[1:])]
            for a, b in itertools.combinations(ms, 2):
                if 0 < b["c"] - a["c"] < a["W"] and colkey(a) != colkey(b):
                    p = pair[(colkey(a), colkey(b))]
                    va, vb = float(a["text"]), float(b["text"])
                    p[0] += 1
                    p[1] += va >= vb
                    p[2] += va <= vb
        self.dscale = {k: max(0.05, float(np.mean(v))) for k, v in dl.items() if len(v) >= 15}
        self.pairs = {}
        for (ka, kb), (n, ge, le) in pair.items():
            if n >= 30 and ge / n >= 0.97 and le / n < 0.97:
                self.pairs[(ka, kb)] = ("ge", ge / n, n)
            elif n >= 30 and le / n >= 0.97 and ge / n < 0.97:
                self.pairs[(ka, kb)] = ("le", le / n, n)
        self.conf = None

    def fit_confusions(self, preds_gts):
        """P(true digit = j | model read digit = i), from equal-length (pred, gt) pairs on the val split."""
        c = np.full((10, 10), 0.1)
        for p, g in preds_gts:
            if len(p) == len(g):
                for a, b in zip(p, g):
                    if a.isdigit() and b.isdigit():
                        c[int(a), int(b)] += 1
        self.conf = c / c.sum(1, keepdims=True)

    def lp_fmt(self, key, s, fam):
        if val(s) is None:
            return BAD
        v = self.fmt.get(key)
        f = fmt(s)
        fam_c = self.fmt_fam[fam]
        pf = (fam_c.get(f, 0) + 0.05) / (sum(fam_c.values()) + 1)
        if not v or sum(v.values()) < 20:
            return math.log(pf)
        return math.log((v.get(f, 0) + 2 * pf) / (sum(v.values()) + 2))

    def lp_rng(self, key, s):
        x = val(s)
        if x is None:
            return BAD
        a = self.vals.get(key)
        if a is None or len(a) < 20:
            return 0.0
        iqr = float(np.subtract(*np.percentile(a, [75, 25])))
        w = max(0.5, iqr / 6)
        cnt = np.searchsorted(a, x + w, "right") - np.searchsorted(a, x - w, "left")
        return math.log((cnt + 0.5) / (len(a) + 1))

    def lp_tmp(self, key, x, nbrs):
        b = self.dscale.get(key)
        if b is None or x is None:
            return 0.0
        t = 0.0
        for y in nbrs:
            d = abs(x - y)
            a = math.log(0.9) - d / b - math.log(2 * b)             # Laplace body ...
            c = math.log(0.1) - d / (10 * b) - math.log(20 * b)    # ... plus a 10x wider outlier component
            t += max(a, c) + math.log1p(math.exp(-abs(a - c)))
        return t

    def lp_pair(self, ka, x, others):
        """others: [(key, value, 'after'|'before')] cells within one table-width of this cell."""
        t = 0.0
        if x is None:
            return t
        for kb, y, rel in others:
            if y is None:
                continue
            c = self.pairs.get((ka, kb) if rel == "after" else (kb, ka))
            if not c:
                continue
            a, b = (x, y) if rel == "after" else (y, x)
            ok = a >= b if c[0] == "ge" else a <= b
            if not ok:
                t += math.log(1 - c[1] + 0.01)
        return t

    def impossible(self, key, s):
        """Unparsable, or outside the min/max of train GT for this column, or a format never seen in this column."""
        x = val(s)
        if x is None:
            return True
        if key not in self.vals or len(self.vals[key]) < 20:
            return False
        return x < self.lo[key] or x > self.hi[key] or self.fmt[key].get(fmt(s), 0) == 0


def candidates(nbest, stats, conf_min=0.02, max_sub=24):
    """nbest: [(string, logp)]. Adds single-digit confusion variants of the top-1 string."""
    cand = dict(nbest)
    if nbest and stats.conf is not None:
        s0, l0 = nbest[0]
        subs = []
        for i, ch in enumerate(s0):
            if not ch.isdigit():
                continue
            for j in range(10):
                if str(j) != ch and stats.conf[int(ch), j] >= conf_min:
                    subs.append((s0[:i] + str(j) + s0[i + 1:],
                                 l0 + math.log(stats.conf[int(ch), j] / stats.conf[int(ch), int(ch)])))
        for s, l in sorted(subs, key=lambda t: -t[1])[:max_sub]:
            if s not in cand or cand[s] < l:
                cand[s] = l
    return cand


def build_context(cells):
    """Per cell: same-column prev/next neighbour, and cells within one table-width (for pair constraints)."""
    pages = collections.defaultdict(list)
    for i, m in enumerate(cells):
        pages[(m["page"], m["table"])].append(i)
    for idxs in pages.values():
        idxs.sort(key=lambda i: cells[i]["c"])
        last, nxt = {}, {}
        for i in idxs:
            k = colkey(cells[i])
            cells[i]["_prev"] = last.get(k)
            last[k] = i
        for i in reversed(idxs):
            k = colkey(cells[i])
            cells[i]["_next"] = nxt.get(k)
            nxt[k] = i
        for i in idxs:
            near = []
            for j in idxs:
                d = cells[j]["c"] - cells[i]["c"]
                if 0 < d < cells[i]["W"]:
                    near.append((j, "after"))
                elif -cells[i]["W"] < d < 0:
                    near.append((j, "before"))
            cells[i]["_near"] = near
    return cells


def rerank(cells, cand_lists, stats, w, margin, passes=2):
    """cand_lists[i] = {string: htr_logp}. Two sweeps: neighbours' values come from the previous sweep."""
    top1 = [max(c, key=c.get) for c in cand_lists]
    cur = list(top1)
    unary = []
    for m, c in zip(cells, cand_lists):
        k = colkey(m)
        unary.append({s: lp + w["fmt"] * stats.lp_fmt(k, s, m["fam"]) + w["rng"] * stats.lp_rng(k, s)
                      for s, lp in c.items()})
    for _ in range(passes):
        new = []
        for i, m in enumerate(cells):
            k = colkey(m)
            nb = [val(cur[j]) for j in (m["_prev"], m["_next"]) if j is not None]
            nb = [x for x in nb if x is not None]
            oth = [(colkey(cells[j]), val(cur[j]), rel) for j, rel in m["_near"]]
            sc = {}
            for s, u in unary[i].items():
                x = val(s)
                if x is None and s != top1[i]:
                    continue                      # an unparsable string must not dodge the temporal/pair penalties
                sc[s] = u + w["tmp"] * stats.lp_tmp(k, x, nb) + w["pair"] * stats.lp_pair(k, x, oth)
            best = max(sc, key=sc.get)
            new.append(best if sc[best] - sc[top1[i]] > margin else top1[i])
        cur = new
    return cur
