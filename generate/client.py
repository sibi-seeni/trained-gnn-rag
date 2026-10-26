import os, json, hashlib
from openai import OpenAI

class CachedLLM:
    def __init__(self, g):
        key = open(os.path.expanduser(g["key_file"])).read().strip()
        self.c, self.g = OpenAI(api_key=key, base_url=g["base_url"]), g
        self.cache = {}
        if os.path.exists(g["cache_file"]):
            for line in open(g["cache_file"]): r = json.loads(line); self.cache[r["k"]] = r["v"]
    def __call__(self, prompt):
        k = hashlib.sha1((self.g["model"] + prompt).encode()).hexdigest()
        if k in self.cache: return self.cache[k], 0.0
        import time; t = time.time()
        r = self.c.chat.completions.create(model=self.g["model"], temperature=self.g["temperature"],
              max_tokens=self.g["max_tokens"], messages=[{"role": "user", "content": prompt}])
        v, dt = r.choices[0].message.content.strip(), time.time() - t
        self.cache[k] = v
        with open(self.g["cache_file"], "a") as f: f.write(json.dumps({"k": k, "v": v}) + "\n")
        return v, dt