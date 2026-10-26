import torch, spacy
from torch_geometric.data import Data
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
KEEP = {"PERSON","ORG","GPE","LOC","WORK_OF_ART","EVENT","NORP","FAC","PRODUCT"}

def build(ex, p_vecs, sim_thresh=None):
    ps = ex["passages"]
    ents = [{e.text.lower() for e in nlp(p["title"] + ". " + p["text"]).ents if e.label_ in KEEP} for p in ps]
    edges, etype = [], []
    for i in range(len(ps)):
        for j in range(len(ps)):
            if i == j: continue
            shared = bool(ents[i] & ents[j])
            title = ps[j]["title"].lower() in ps[i]["text"].lower()
            sim = sim_thresh is not None and float(p_vecs[i] @ p_vecs[j]) > sim_thresh
            if shared or title or sim:
                edges.append((i, j)); etype.append([shared, title, sim])
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty(2, 0, dtype=torch.long)
    ea = torch.tensor(etype, dtype=torch.float) if etype else torch.empty(0, 3)
    ea = torch.cat([ea, torch.zeros(ea.size(0), 2)], dim=1)  # cols 3-4 reserved: agree/contradict (Week 3)
    return Data(x=torch.tensor(p_vecs, dtype=torch.float), edge_index=ei, edge_attr=ea,
                y=torch.tensor([p["gold"] for p in ps], dtype=torch.float), qid=ex["qid"])