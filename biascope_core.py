import bisect, html, json, os, re, string, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

BASE_DIR = Path("/content/drive/MyDrive/BIASCOPE")
ADAPTER_DIR = BASE_DIR / "models" / "final" / "xlmr_lora_mbib_babe_s42"
TEMP_JSON = BASE_DIR / "models" / "final" / "final_temperature.json"
MAX_LEN = 160
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEV_RE = re.compile(r"[\u0900-\u097F]")
LANG_NAME = {"en": "English", "hi": "Hindi (Devanagari script)"}

def detect_lang(text):
    dev = len(DEV_RE.findall(text))
    lat = len(re.findall(r"[A-Za-z]", text))
    return "hi" if dev > lat else "en"

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def word_scores(text, attr, offsets):
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    starts = [s for s, _ in spans]
    scores = np.zeros(len(spans))
    for a, (s, e) in zip(attr, offsets):
        if e <= s:
            continue
        pos = max(s, e - 1)
        k = bisect.bisect_right(starts, pos) - 1
        if k >= 0 and pos < spans[k][1]:
            scores[k] += a
    return [text[s:e] for s, e in spans], scores

def render_html(words, scores):
    mx = max(float(np.max(np.abs(scores))) if len(scores) else 0.0, 1e-9)
    parts = []
    for w, s in zip(words, scores):
        a = min(abs(float(s)) / mx, 1.0)
        color = "255,80,80" if s > 0 else "70,130,255"
        style = "background: rgba(%s,%.2f); padding:1px 3px; border-radius:3px; margin:1px;" % (color, 0.05 + 0.75 * a)
        parts.append('<span style="%s">%s</span>' % (style, html.escape(w)))
    legend = ('<div style="font-size:12px;margin-bottom:6px">'
              '<span style="background:rgba(255,80,80,.6);padding:1px 4px">red = biased ki taraf</span> '
              '<span style="background:rgba(70,130,255,.6);padding:1px 4px">blue = neutral ki taraf</span></div>')
    return legend + '<div style="font-size:17px;line-height:2.1">' + " ".join(parts) + "</div>"

class BiasModel:
    def __init__(self, adapter_dir=ADAPTER_DIR, temp_json=TEMP_JSON):
        adapter_dir = Path(adapter_dir)
        assert adapter_dir.exists(), "Adapter nahi mila: %s" % adapter_dir
        self.tok = AutoTokenizer.from_pretrained(str(adapter_dir))
        assert self.tok.is_fast, "Fast tokenizer chahiye"
        base_name = PeftConfig.from_pretrained(str(adapter_dir)).base_model_name_or_path
        base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
        self.model = PeftModel.from_pretrained(base, str(adapter_dir)).to(DEVICE).eval()
        self.emb = self.model.base_model.model.get_input_embeddings()
        self.T = 1.0
        if Path(temp_json).exists():
            self.T = float(json.load(open(temp_json))["temperature"])

    @torch.no_grad()
    def margins(self, texts, bs=32):
        texts = [str(t) for t in texts]
        if not texts:
            return np.array([])
        order = np.argsort([len(t) for t in texts])
        out = np.zeros(len(texts))
        for b in range(0, len(texts), bs):
            idx = order[b:b + bs]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_LEN).to(DEVICE)
            lg = self.model(**enc).logits.float()
            out[idx] = (lg[:, 1] - lg[:, 0]).cpu().numpy()
        return out

    def probs(self, texts, calibrated=True):
        return sigmoid(self.margins(texts) / (self.T if calibrated else 1.0))

    def explain(self, text, steps=32, top_k=5):
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        ids, mask = enc["input_ids"], enc["attention_mask"]
        with torch.no_grad():
            x = self.emb(ids)
            base_ids = torch.full_like(ids, self.tok.pad_token_id)
            base_ids[0, 0], base_ids[0, -1] = self.tok.cls_token_id, self.tok.sep_token_id
            b = self.emb(base_ids)
        total = torch.zeros_like(x)
        for k in range(1, steps + 1):
            e = (b + (k / steps) * (x - b)).detach().requires_grad_(True)
            lg = self.model(inputs_embeds=e, attention_mask=mask).logits[0]
            g, = torch.autograd.grad(lg[1] - lg[0], e)
            total += g
        attr = ((x - b) * total / steps).sum(-1)[0].detach().cpu().numpy()
        words, scores = word_scores(text, attr, offsets)
        top = [(words[i], float(scores[i])) for i in np.argsort(-scores)[:top_k]]
        p_cal = float(self.probs([text])[0])
        return dict(words=words, scores=scores, top=top, p_cal=p_cal)

def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        try:
            from google.colab import userdata
            key = userdata.get("GEMINI_API_KEY")
        except Exception:
            key = None
    if not key:
        import getpass
        key = getpass.getpass("Gemini API key (Colab Secrets me GEMINI_API_KEY rakhna behtar hai): ")
    return key

class Gemini:
    PREFERRED = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.0-flash"]

    def __init__(self, api_key=None, model=None, min_interval=2.0):
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=api_key or get_api_key())
        self.min_interval = min_interval
        self._last = 0.0
        self.available = self._list_models()
        self.model = model or self._pick()

    def _list_models(self):
        names = []
        try:
            for m in self.client.models.list():
                acts = getattr(m, "supported_actions", None) or []
                if (not acts) or ("generateContent" in acts):
                    names.append(m.name.replace("models/", ""))
        except Exception as e:
            print("model list nahi mili:", repr(e)[:150])
        return names

    def _pick(self):
        for p in self.PREFERRED:
            if p in self.available:
                return p
        bad = ("lite", "image", "tts", "live", "audio", "embedding", "robotics", "computer")
        flash = [n for n in self.available if "flash" in n and not any(b in n for b in bad)]
        if flash:
            return sorted(flash)[-1]
        return self.PREFERRED[0]

    def ask(self, prompt, json_mode=True, retries=6):
        for attempt in range(retries):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                cfg = self.types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                self._last = time.time()
                txt = getattr(resp, "text", None)
                return txt if txt else None
            except Exception as e:
                self._last = time.time()
                s = str(e)
                transient = any(k in s for k in ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE", "overloaded"))
                if (not transient) or attempt == retries - 1:
                    raise
                time.sleep(min(60, 4 * 2 ** attempt))

    def ask_field(self, prompt, field):
        txt = self.ask(prompt, json_mode=True)
        if txt is None:
            return None
        t = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
        try:
            obj = json.loads(t)
            if isinstance(obj, list):
                obj = obj[0] if obj else {}
            v = obj.get(field) if isinstance(obj, dict) else None
            return str(v).strip() if v else None
        except Exception:
            m = re.search(r'"%s"\s*:\s*"(.*?)"\s*[,}]' % re.escape(field), t, flags=re.S)
            return m.group(1).strip() if m else None

def explain_prompt(text, p, top_words, lang):
    return ("You explain the output of a media-bias classifier. The classifier, not you, decided the label.\n"
            "Text (" + LANG_NAME[lang] + "): " + json.dumps(text, ensure_ascii=False) + "\n"
            "Classifier: calibrated probability that the text is biased = " + format(p, ".2f") + "\n"
            "Words that most influenced the score (Integrated Gradients): " + ", ".join(top_words) + "\n"
            "Task: in 2-3 sentences, explain why the text reads as biased (or as mostly neutral if the probability is "
            "below 0.5), citing the influential words as evidence. Do not state facts that are not in the text. "
            "Write the explanation in " + LANG_NAME[lang] + ".\n"
            'Return JSON: {"explanation": "..."}')

def rewrite_prompt(text, lang):
    return ("Rewrite the text below in a neutral, factual tone.\n"
            "Rules: keep the same language (" + LANG_NAME[lang] + "); keep every factual claim, name and number; "
            "remove or replace loaded words, insults, sensational or clickbait phrasing, speculation or opinion "
            "presented as fact; keep the length similar; do not add new facts.\n"
            "Text: " + json.dumps(text, ensure_ascii=False) + "\n"
            'Return JSON: {"rewrite": "..."}')

def paraphrase_prompt(text, lang):
    return ("Paraphrase the text below with different wording but keep its meaning, tone and level of emotional or "
            "loaded language exactly the same. Do NOT make it more neutral. Keep the same language (" + LANG_NAME[lang] + ").\n"
            "Text: " + json.dumps(text, ensure_ascii=False) + "\n"
            'Return JSON: {"paraphrase": "..."}')
