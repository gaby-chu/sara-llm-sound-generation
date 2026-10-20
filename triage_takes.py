"""Triage every take with a Gemini audio-input model, then pick the best take per clip. Rough filter, not a scorer."""
import csv, json, re, sys, time
from pathlib import Path
from google import genai
from google.genai import types
c = genai.Client(vertexai=True, project="capstone-510119", location="us-central1")
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")  # run folder holding clips/ (default: here)
LABELS = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).parent / "labels_v2.csv"
Q = ("This is a 2.5 second clip of an isolated speech sound for a speech therapy dataset. Reply with JSON only: "
     '{"heard_ipa": "the single main sound you hear, IPA", "vowel_after": true/false, "letter_name_or_word": true/false, '
     '"voiced": true/false/null, "sustained_or_burst": "sustained|burst|unclear", "child_like_voice": true/false, "notes": "max 12 words"}')
EQ = {"ɹ":"ɹ","r":"ɹ","ɚ":"ɹ","ɝ":"ɹ","ɡ":"g"}
def norm(x): return EQ.get(re.sub(r"[/ːˑ̞ ]","",x or "")[:1],re.sub(r"[/ːˑ̞ ]","",x or "")[:1])
OUT = ROOT / "triage_takes.json"
res = json.load(open(OUT)) if OUT.exists() else {}
rows = list(csv.DictReader(open(LABELS, encoding="utf-8")))
for r in rows:
    stem = r["file"][:-4]
    for f in sorted((ROOT / "clips").glob(f"{stem}_take*.wav")):
        if f.name in res: continue
        for attempt in range(6):
            try:
                resp = c.models.generate_content(model="gemini-2.5-flash",
                    contents=[types.Part.from_bytes(data=f.read_bytes(), mime_type="audio/wav"), Q],
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0))
                break
            except Exception as e:
                print("retry", str(e)[:50], file=sys.stderr); time.sleep(15 * (attempt + 1))
        else: continue
        try: d = json.loads(resp.text)
        except Exception: d = {}
        res[f.name] = d; time.sleep(2)
        print(f.name, "want", r["produced_IPA"] or "-", "heard", d.get("heard_ipa"), "vowel", d.get("vowel_after"), "name", d.get("letter_name_or_word"), flush=True)
        json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
