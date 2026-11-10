"""Plant fake passages into the passage graphs (Week 3 / Week 5 attacks).

Usage (from repo root):
    python -m attacks.plant configs/trust_stage.yaml --split val   --attack none
    python -m attacks.plant configs/trust_stage.yaml --split val   --attack edit
    python -m attacks.plant configs/trust_stage.yaml --split train --attack edit       # training fakes
    python -m attacks.plant configs/trust_stage.yaml --split val   --attack connected  # Week 5
    python -m attacks.plant configs/trust_stage.yaml --split val   --attack connected_q

Writes data/processed/{split}_{attack}.jsonl and data/processed/graphs_{split}_{attack}.pt.
`--attack none` is the clean set rebuilt with the Week 3 edge columns (agree / contradict), so clean and attacked
graphs come from the same builder.

Attacks (all fakes are edits of real passages; 3 fakes per question; fixed seed per split):
  edit         copy a gold passage and swap entities in it (never the article's own subject, i.e. anything in the title) (the answer first, if the passage contains it) for
               other entities of the same type taken from other questions. Title kept, so the fake looks like
               the same article and picks up the same title-mention edges. Sources: hop2 gold, hop1 gold, hop2
               gold again (a different swap).
  connected    like edit, but the non-first sentences are shuffled (so it is no longer a near copy) and a
               "See also" sentence naming 3 other passages of the question is appended, so the fake has title
               edges into the rest of the graph (well-connected fake).
  connected_q  connected + the question text prepended (PoisonedRAG-style), which pushes its similarity to the
               question up. Hardest of the three for a relevance ranker.

Fakes are appended after the real passages (node indices 10, 11, 12 for a 10-passage question) and carry
`fake: true`, `gold: false`, `src: <index of the passage they were made from>`.

Edge attribute columns (same layout as graph/build_graphs.py):
    0 shared_entity   1 title_mention   2 sim (unused, 0)   3 agree   4 contradict
  overlap(i, j)  = same title, or word-set Jaccard >= 0.5
  contradict     = overlap and the two passages' entity sets differ
  agree          = not contradict, and they share >= 2 entities (any type, incl. dates and numbers)
Cols 0-1 are computed exactly like graph/build_graphs.py; `--check` verifies that on the clean graphs.
"""
import argparse, json, os, random, re

import numpy as np
import spacy
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from common.config import load_cfg
from graph.dataset import hop2_index
from retrieval.embed import encode_cached

KEEP = {"PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "EVENT", "NORP", "FAC", "PRODUCT"}  # = build_graphs.KEEP
SWAP = KEEP | {"DATE", "CARDINAL", "QUANTITY", "MONEY", "PERCENT", "ORDINAL", "TIME", "LANGUAGE", "LAW"}
SEEDS = {"train": 8, "val": 7}  # fake seeds per split, fixed
N_FAKES = 3
WORD = re.compile(r"[a-z0-9]+")
CLEAN = re.compile(r"^[A-Z0-9][\w.'&-]*( [\w.'&-]+)*$")  # pool entries: start upper-case/digit, plain tokens
_nlp = None


def nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _nlp


def doc_text(p):
    return p["title"] + ". " + p["text"]


def ner(texts, bs=256):
    return [[(e.text, e.label_) for e in d.ents] for d in nlp().pipe(texts, batch_size=bs)]


def ner_cached(exs, split, out_dir):
    path = f"{out_dir}/ents_{split}.json"
    if os.path.exists(path):
        cached = json.load(open(path))
        if len(cached) == len(exs):
            return [[[tuple(e) for e in ps] for ps in q] for q in cached]
    flat = ner([doc_text(p) for e in exs for p in e["passages"]])
    out, i = [], 0
    for e in exs:
        out.append(flat[i:i + len(e["passages"])]); i += len(e["passages"])
    json.dump(out, open(path, "w"))
    return out


# ---------------------------------------------------------------- edges

def words(p):
    return set(WORD.findall(p["text"].lower()))


def edge_tensors(ps, ents):
    """Directed edges i->j with 5 columns, built the same way as graph/build_graphs.build (cols 0-1)."""
    keep = [{t.lower() for t, l in es if l in KEEP} for es in ents]
    alle = [{t.lower() for t, l in es if l in SWAP} for es in ents]
    ws = [words(p) for p in ps]
    edges, attr = [], []
    for i in range(len(ps)):
        for j in range(len(ps)):
            if i == j:
                continue
            shared = bool(keep[i] & keep[j])
            title = ps[j]["title"].lower() in ps[i]["text"].lower()
            jac = len(ws[i] & ws[j]) / max(1, len(ws[i] | ws[j]))
            overlap = ps[i]["title"].lower() == ps[j]["title"].lower() or jac >= 0.5
            contra = overlap and alle[i] != alle[j]
            agree = (not contra) and len(alle[i] & alle[j]) >= 2
            if shared or title or contra or agree:
                edges.append((i, j)); attr.append([shared, title, False, agree, contra])
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty(2, 0, dtype=torch.long)
    ea = torch.tensor(attr, dtype=torch.float) if attr else torch.empty(0, 5)
    return ei, ea


# ---------------------------------------------------------------- fakes

def entity_pool(all_ents):
    pool = {}
    for q in all_ents:
        for es in q:
            for t, l in es:
                if l in SWAP and 1 < len(t) < 40 and CLEAN.match(t) and len(t.split()) <= 4:
                    pool.setdefault(l, set()).add(t)
    return {l: sorted(v) for l, v in pool.items()}


def swap_entities(p, ents, answer, own, pool, rng, n_extra=1):
    """Return an edited copy of passage p's text with 1-2 entities swapped for same-type ones."""
    text = p["text"]
    cands = [(t, l) for t, l in dict.fromkeys(ents) if l in SWAP and t in text and t.lower() not in p["title"].lower()]
    if not cands:
        return None
    chosen = []
    ans = [(t, l) for t, l in cands if t.lower() == answer.lower() or (len(answer) > 3 and answer.lower() in t.lower())]
    if ans:
        chosen.append(ans[0])
    rest = [c for c in cands if c not in chosen]
    rng.shuffle(rest)
    chosen += rest[: (n_extra if chosen else n_extra + 1)]
    for t, l in chosen:
        options = [x for x in pool.get(l, []) if x.lower() not in own]
        if not options:
            continue
        text = text.replace(t, rng.choice(options))
    return text if text != p["text"] else None


def make_fakes(ex, ents, cos, pool, rng, attack):
    ps = ex["passages"]
    gold = [j for j, p in enumerate(ps) if p["gold"]]
    h2, _ = hop2_index(ex, cos)
    if h2 < 0 or len(gold) != 2:
        return []
    h1 = gold[0] if gold[1] == h2 else gold[1]
    own = {t.lower() for es in ents for t, _ in es}
    fakes = []
    for n, src in enumerate([h2, h1, h2][:N_FAKES]):
        p = ps[src]
        text = None
        for _ in range(5):  # retry a few times so the two hop2 fakes differ
            text = swap_entities(p, ents[src], ex["answer"], own, pool, rng)
            if text and all(text != f["text"] for f in fakes):
                break
        if text is None:  # no swappable entity: fall back to reversing the sentence order
            sents = re.split(r"(?<=[.!?])\s+", p["text"])
            text = " ".join(sents[::-1]) if len(sents) > 1 else p["text"] + " "
        if attack in ("connected", "connected_q"):
            sents = re.split(r"(?<=[.!?])\s+", text)
            head, tail = sents[:1], sents[1:]
            rng.shuffle(tail)
            others = [j for j in np.argsort(-cos) if j != src][:3]
            text = " ".join(head + tail) + " See also: " + ", ".join(ps[j]["title"] for j in others) + "."
            if attack == "connected_q":
                text = ex["question"] + " " + text
        fakes.append({"pid": f'{ex["qid"]}_fake{n}', "title": p["title"], "text": text,
                      "gold": False, "fake": True, "src": int(src)})
    return fakes


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--split", default="val")
    ap.add_argument("--attack", default="edit", choices=["none", "edit", "connected", "connected_q"])
    ap.add_argument("--check", action="store_true", help="check cols 0-1 against the saved Week 1 graphs")
    a = ap.parse_args()

    cfg = load_cfg(a.config)
    out_dir, E = cfg["data"]["out_dir"], cfg["embed"]
    exs = [json.loads(l) for l in open(f"{out_dir}/{a.split}.jsonl")]
    ents = ner_cached(exs, a.split, out_dir)
    offs = np.cumsum([0] + [len(e["passages"]) for e in exs])
    pv = encode_cached([doc_text(p) for e in exs for p in e["passages"]], E["model"], E["cache_dir"], f"p_{a.split}_all")
    qv = encode_cached([e["question"] for e in exs], E["model"], E["cache_dir"], f"q_{a.split}_all")

    rng = random.Random(SEEDS[a.split] * 1000 + ["none", "edit", "connected", "connected_q"].index(a.attack))
    pool = entity_pool(ents)
    new_exs, new_ents = [], []
    for i, ex in enumerate(tqdm(exs, desc=f"fakes {a.split}/{a.attack}")):
        cos = pv[offs[i]:offs[i + 1]] @ qv[i]
        fakes = [] if a.attack == "none" else make_fakes(ex, ents[i], cos, pool, rng, a.attack)
        new_exs.append({**ex, "passages": [{**p, "fake": False} for p in ex["passages"]] + fakes})
        new_ents.append(ents[i])
    # NER + embeddings for the fakes only
    fake_list = [(i, p) for i, e in enumerate(new_exs) for p in e["passages"] if p["fake"]]
    if fake_list:
        fe = ner([doc_text(p) for _, p in fake_list])
        fv = encode_cached([doc_text(p) for _, p in fake_list], E["model"], E["cache_dir"], f"fake_{a.split}_{a.attack}")
        for (i, _), es in zip(fake_list, fe):
            new_ents[i] = new_ents[i] + [es]
    graphs, k = [], 0
    for i, ex in enumerate(tqdm(new_exs, desc="graphs")):
        n_f = sum(p["fake"] for p in ex["passages"])
        vec = pv[offs[i]:offs[i + 1]]
        if n_f:
            vec = np.concatenate([vec, fv[k:k + n_f]]); k += n_f
        ei, ea = edge_tensors(ex["passages"], new_ents[i])
        graphs.append(Data(x=torch.tensor(vec, dtype=torch.float), edge_index=ei, edge_attr=ea,
                           y=torch.tensor([p["gold"] for p in ex["passages"]], dtype=torch.float),
                           fake=torch.tensor([p["fake"] for p in ex["passages"]], dtype=torch.bool), qid=ex["qid"]))

    tag = f"{a.split}_{a.attack}"
    with open(f"{out_dir}/{tag}.jsonl", "w") as f:
        for r in new_exs:
            f.write(json.dumps(r) + "\n")
    torch.save(graphs, f"{out_dir}/graphs_{tag}.pt")
    n_f = [int(g.fake.sum()) for g in graphs]
    n_contra = [int(g.edge_attr[:, 4].sum()) // 2 for g in graphs]
    print(json.dumps({"set": tag, "n": len(graphs), "fakes_per_q": float(np.mean(n_f)),
                      "avg_edges": float(np.mean([g.edge_index.size(1) for g in graphs])),
                      "contradict_pairs_per_q": float(np.mean(n_contra)),
                      "agree_edges_per_q": float(np.mean([g.edge_attr[:, 3].sum().item() for g in graphs]))}))

    if a.check:
        old = torch.load(f"{out_dir}/graphs_{a.split}.pt", weights_only=False)
        bad = 0
        for g, o in zip(graphs, old):
            m = ~g.fake[g.edge_index].any(0) & (g.edge_attr[:, :2].sum(1) > 0)
            new = {(int(s), int(t), int(x), int(y)) for (s, t), (x, y) in zip(g.edge_index[:, m].t().tolist(), g.edge_attr[m, :2].tolist())}
            ref = {(int(s), int(t), int(x), int(y)) for (s, t), (x, y) in zip(o.edge_index.t().tolist(), o.edge_attr[:, :2].tolist())}
            bad += new != ref
        print(f"check vs saved graphs_{a.split}.pt (cols 0-1, real passages): {bad} of {len(graphs)} graphs differ")


if __name__ == "__main__":
    main()
