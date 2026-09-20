#!/usr/bin/env bash
# Score GSM8K against a running arm and record the result.
#   usage: pit2_gsm8k.sh <arm> [node] [limit]
set -u
ARM="${1:?arm}"
NODE="${2:-pit2-p03-g02}"
LIMIT="${3:-200}"

timeout 5400 ssh -o BatchMode=yes -o ConnectTimeout=15 "$NODE" "ARM='$ARM' LIMIT='$LIMIT' bash -s" <<'REMOTE'
set -u
RUN=$(readlink -f "/home/$(whoami)/k3ab/runs/current-$ARM")
echo "RUN=$RUN  ARM=$ARM  LIMIT=$LIMIT"

code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://127.0.0.1:8000/health || echo 000)
if [ "$code" != "200" ]; then echo "server not healthy (health=$code); aborting"; exit 3; fi

cat > "$RUN/gsm8k.py" <<'PY'
"""Minimal 5-shot GSM8K scorer against an OpenAI-compatible completions API.

Deliberately dependency-light: the stock vLLM image has no lm_eval, and the
point here is a self-consistent number across three attention arms, not a
leaderboard-comparable one. Same prompts, same decode params, same scorer for
every arm.
"""
import json, os, re, sys, urllib.request, concurrent.futures as cf

BASE = os.environ.get("BASE", "http://127.0.0.1:8000/v1/completions")
MODEL = os.environ.get("MODEL", "Kimi-K3")
LIMIT = int(os.environ.get("LIMIT", "200"))
CONC = int(os.environ.get("CONC", "16"))
DATA = os.environ.get("GSM8K_JSONL", "/run_out/gsm8k_test.jsonl")

rows = [json.loads(l) for l in open(DATA)][:LIMIT]

FEWSHOT = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether. The answer is 72.

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $0.2 per minute. For 50 minutes, she earned 0.2 x 50 = $10. The answer is 10.

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: Betty has 100/2 = $50. Her grandparents gave her 15*2 = $30. In total she has 50+15+30 = $95. She needs 100-95 = $5 more. The answer is 5.

Question: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
Answer: Today Julie read 12*2 = 24 pages. So far she read 12+24 = 36 pages. There are 120-36 = 84 pages remaining. Half of that is 84/2 = 42. The answer is 42.

Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Answer: Each time he writes 3*2 = 6 pages. Twice a week that is 6*2 = 12 pages. In a year that is 12*52 = 624 pages. The answer is 624.

"""

NUM = re.compile(r"-?\d[\d,]*\.?\d*")

def gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")

def pred(text):
    seg = text.split("Question:")[0]
    m = re.search(r"answer is\s*(-?[\d,\.]+)", seg, re.I)
    cand = m.group(1) if m else (NUM.findall(seg) or [""])[-1]
    return cand.strip().rstrip(".").replace(",", "")

def norm(x):
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return x

def ask(row):
    body = json.dumps({
        "model": MODEL,
        "prompt": FEWSHOT + "Question: " + row["question"] + "\nAnswer:",
        "max_tokens": 320,
        "temperature": 0.0,
        "stop": ["Question:"],
    }).encode()
    req = urllib.request.Request(BASE, data=body,
                                headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                out = json.load(r)["choices"][0]["text"]
            return norm(pred(out)) == norm(gold(row["answer"])), out
        except Exception as e:
            err = repr(e)
    return False, "ERR " + err

ok = 0
fails = []
with cf.ThreadPoolExecutor(max_workers=CONC) as ex:
    for i, (good, text) in enumerate(ex.map(ask, rows)):
        ok += good
        if not good and len(fails) < 5:
            fails.append(text[:300])
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}  acc={ok/(i+1):.4f}", flush=True)

acc = ok / len(rows)
print(json.dumps({"arm": os.environ.get("ARM", "?"), "n": len(rows),
                  "correct": ok, "accuracy": round(acc, 4)}))
with open("/run_out/gsm8k_result.json", "w") as f:
    json.dump({"arm": os.environ.get("ARM", "?"), "n": len(rows),
               "correct": ok, "accuracy": acc}, f)
for t in fails:
    print("--- miss sample ---"); print(t)
PY

# Fetch the GSM8K test split once into the shared run dir.
if [ ! -s "$RUN/gsm8k_test.jsonl" ]; then
  CACHE="/home/$(whoami)/k3ab/gsm8k_test.jsonl"
  if [ -s "$CACHE" ]; then
    cp "$CACHE" "$RUN/gsm8k_test.jsonl"
  else
    echo "downloading gsm8k test split"
    curl -sL -o "$CACHE" \
      https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
      && cp "$CACHE" "$RUN/gsm8k_test.jsonl" || { echo "download failed"; exit 4; }
  fi
fi
echo "gsm8k rows: $(wc -l < "$RUN/gsm8k_test.jsonl")"

docker exec -e ARM="$ARM" -e LIMIT="$LIMIT" -e CONC=16 "k3ab-$ARM" \
  python3 /run_out/gsm8k.py 2>&1 | tail -25
REMOTE
