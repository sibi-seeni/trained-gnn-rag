import json, random, os
from datasets import load_dataset

def run(cfg):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
    random.seed(cfg["seed"])
    os.makedirs(cfg["data"]["out_dir"], exist_ok=True)
    for split, n in [("train", cfg["data"]["n_train"]), ("validation", cfg["data"]["n_val"])]:
        idx = random.sample(range(len(ds[split])), n)
        out = []
        for i in idx:
            ex = ds[split][i]
            gold = set(ex["supporting_facts"]["title"])
            passages = [{"pid": f'{ex["id"]}_{j}', "title": t, "text": "".join(s), "gold": t in gold}
                        for j, (t, s) in enumerate(zip(ex["context"]["title"], ex["context"]["sentences"]))]
            out.append({"qid": ex["id"], "question": ex["question"], "answer": ex["answer"],
                        "type": ex["type"], "passages": passages})
        name = "val" if split == "validation" else "train"
        with open(f'{cfg["data"]["out_dir"]}/{name}.jsonl', "w") as f:
            for r in out: f.write(json.dumps(r) + "\n")