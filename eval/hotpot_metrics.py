import re, string
from collections import Counter
def norm(s):
    s = s.lower(); s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def em(p, g): return float(norm(p) == norm(g))
def f1(p, g):
    pn, gn = norm(p), norm(g)
    if pn in {"yes","no","noanswer"} or gn in {"yes","no","noanswer"}:
        if pn != gn: return 0.0
    pt, gt = pn.split(), gn.split()
    c = sum((Counter(pt) & Counter(gt)).values())
    if c == 0: return 0.0
    pr, rc = c/len(pt), c/len(gt); return 2*pr*rc/(pr+rc)