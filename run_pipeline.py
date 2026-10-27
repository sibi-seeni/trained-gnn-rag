import yaml, json, sys, os, time, numpy as np
from data.load_hotpot import run as load
from retrieval.embed import encode_cached
from retrieval.dense import rank
from eval.retrieval_metrics import retrieval_metrics
from eval.hotpot_metrics import em, f1
from generate.client import CachedLLM
from generate.answer import build_prompt

cfg = yaml.safe_load(open(sys.argv[1]))
if not os.path.exists(f'{cfg["data"]["out_dir"]}/val.jsonl'): load(cfg)
exs = [json.loads(l) for l in open(f'{cfg["data"]["out_dir"]}/val.jsonl')][:cfg["eval"]["n_questions"]]

E = cfg["embed"]
qv = encode_cached([e["question"] for e in exs], E["model"], E["cache_dir"], "q_val")
pv = encode_cached([p["title"]+". "+p["text"] for e in exs for p in e["passages"]], E["model"], E["cache_dir"], "p_val")
offs = np.cumsum([0] + [len(e["passages"]) for e in exs])

llm, k = CachedLLM(cfg["generate"]), cfg["retrieval"]["k"]
flags, ems, f1s, lat = [], [], [], []
for i, e in enumerate(exs):
    P = pv[offs[i]:offs[i+1]]
    t = time.time(); order = rank(qv[i], P); r_lat = time.time() - t
    flags.append([e["passages"][j]["gold"] for j in order])
    pred, g_lat = llm(build_prompt(e["question"], [e["passages"][j] for j in order[:k]]))
    ems.append(em(pred, e["answer"])); f1s.append(f1(pred, e["answer"])); lat.append(r_lat + g_lat)

row = {"method": cfg["name"], "EM": np.mean(ems), "F1": np.mean(f1s),
       **retrieval_metrics(flags, cfg["retrieval"]["ks"]), "latency_s": np.mean(lat)}
print(json.dumps(row, indent=2))
os.makedirs("results", exist_ok=True)
with open("results/results.jsonl", "a") as f: f.write(json.dumps(row) + "\n")