from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="PhishGuard API — Quad Model Protection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ── Load DNN ──────────────────────────────────────────────────
dnn_model = tf.keras.models.load_model("phishing_model_local.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("✅ DNN Model loaded!")

# ── Load NLP ──────────────────────────────────────────────────
nlp_model = tf.keras.models.load_model("phishguard_nlp_model.h5")
with open("nlp_tokenizer.pkl", "rb") as f:
    nlp_tokenizer = pickle.load(f)
with open("nlp_label_map.pkl", "rb") as f:
    nlp_label_map = pickle.load(f)
NLP_MAX_LEN = 100
print("✅ NLP Model loaded!")

# ── Load RNN ──────────────────────────────────────────────────
rnn_model = tf.keras.models.load_model("phishguard_rnn_model.h5")
with open("rnn_char2idx.pkl", "rb") as f:
    rnn_char2idx = pickle.load(f)
RNN_MAX_LEN = 200
print("✅ RNN Model loaded!")

# ── Load CNN ──────────────────────────────────────────────────
cnn_model = tf.keras.models.load_model("phishguard_cnn_model.h5")
with open("cnn_char2idx.pkl", "rb") as f:
    cnn_char2idx = pickle.load(f)
CNN_MAX_LEN = 200
print("✅ CNN Model loaded!")

print("\n🛡️ PhishGuard API — All 4 models ready!")

# Thread pool for parallel model inference
executor = ThreadPoolExecutor(max_workers=4)

# ── Schemas ───────────────────────────────────────────────────
class WebsiteFeatures(BaseModel):
    UsingIP: int
    LongURL: int
    ShortURL: int
    Symbol: int
    Redirecting: int
    PrefixSuffix: int
    SubDomains: int
    HTTPS: int
    DomainRegLen: int
    Favicon: int
    NonStdPort: int
    HTTPSDomainURL: int
    RequestURL: int
    AnchorURL: int
    LinksInScriptTags: int
    ServerFormHandler: int
    InfoEmail: int
    AbnormalURL: int
    WebsiteForwarding: int
    StatusBarCust: int
    DisableRightClick: int
    UsingPopupWindow: int
    IframeRedirection: int
    AgeofDomain: int
    DNSRecording: int
    WebsiteTraffic: int
    PageRank: int
    GoogleIndex: int
    LinksPointingToPage: int
    StatsReport: int

class TextInput(BaseModel):
    text: str

class URLInput(BaseModel):
    url: str

class CombinedInput(BaseModel):
    url: str
    text: Optional[str] = None
    features: Optional[WebsiteFeatures] = None

# ── Home ──────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "status": "PhishGuard API is running!",
        "models": {
            "DNN": "Website feature detection — 96.72% accuracy",
            "NLP": "Email/SMS text detection — 98.32% accuracy",
            "RNN": "URL character detection — 95.88% accuracy",
            "CNN": "URL pattern detection — 97.15% accuracy"
        },
        "endpoints": [
            "/predict/dnn",
            "/predict/nlp",
            "/predict/rnn",
            "/predict/cnn",
            "/predict/combined"
        ]
    }

# ── DNN Endpoint ──────────────────────────────────────────────
@app.post("/predict/dnn")
def predict_dnn(features: WebsiteFeatures):
    data = np.array([[
        features.UsingIP, features.LongURL, features.ShortURL,
        features.Symbol, features.Redirecting, features.PrefixSuffix,
        features.SubDomains, features.HTTPS, features.DomainRegLen,
        features.Favicon, features.NonStdPort, features.HTTPSDomainURL,
        features.RequestURL, features.AnchorURL, features.LinksInScriptTags,
        features.ServerFormHandler, features.InfoEmail, features.AbnormalURL,
        features.WebsiteForwarding, features.StatusBarCust,
        features.DisableRightClick, features.UsingPopupWindow,
        features.IframeRedirection, features.AgeofDomain,
        features.DNSRecording, features.WebsiteTraffic, features.PageRank,
        features.GoogleIndex, features.LinksPointingToPage, features.StatsReport
    ]])
    data_scaled = scaler.transform(data)
    prediction = dnn_model.predict(data_scaled, verbose=0)[0][0]
    return {
        "model": "DNN",
        "prediction": "Legitimate" if prediction > 0.5 else "Phishing",
        "confidence": round(float(prediction) * 100, 2)
    }

# ── NLP Endpoint ──────────────────────────────────────────────
@app.post("/predict/nlp")
def predict_nlp(input: TextInput):
    sequence = nlp_tokenizer.texts_to_sequences([input.text])
    padded = pad_sequences(sequence, maxlen=NLP_MAX_LEN, padding='post')
    prediction = nlp_model.predict(padded, verbose=0)[0][0]
    is_phishing = prediction > 0.5
    confidence = prediction * 100 if is_phishing else (1 - prediction) * 100
    return {
        "model": "NLP",
        "prediction": "Phishing" if is_phishing else "Legitimate",
        "confidence": round(float(confidence), 2)
    }

# ── RNN Endpoint ──────────────────────────────────────────────
@app.post("/predict/rnn")
def predict_rnn(input: URLInput):
    sequence = [rnn_char2idx.get(c, 1) for c in str(input.url)[:RNN_MAX_LEN]]
    padded = pad_sequences([sequence], maxlen=RNN_MAX_LEN, padding='post')
    prediction = rnn_model.predict(padded, verbose=0)[0][0]
    is_phishing = prediction > 0.5
    confidence = prediction * 100 if is_phishing else (1 - prediction) * 100
    return {
        "model": "RNN",
        "prediction": "Phishing" if is_phishing else "Legitimate",
        "confidence": round(float(confidence), 2)
    }

# ── CNN Endpoint ──────────────────────────────────────────────
@app.post("/predict/cnn")
def predict_cnn(input: URLInput):
    sequence = [cnn_char2idx.get(c, 1) for c in str(input.url)[:CNN_MAX_LEN]]
    padded = pad_sequences([sequence], maxlen=CNN_MAX_LEN, padding='post')
    prediction = cnn_model.predict(padded, verbose=0)[0][0]
    is_phishing = prediction > 0.5
    confidence = prediction * 100 if is_phishing else (1 - prediction) * 100
    return {
        "model": "CNN",
        "prediction": "Phishing" if is_phishing else "Legitimate",
        "confidence": round(float(confidence), 2)
    }

# ── Combined Endpoint (Parallel) ──────────────────────────────
@app.post("/predict/combined")
async def predict_combined(input: CombinedInput):
    loop = asyncio.get_event_loop()
    results = []
    phishing_votes = 0

    # ── Model runner functions ────────────────────────────────
    def run_rnn():
        sequence = [rnn_char2idx.get(c, 1) for c in str(input.url)[:RNN_MAX_LEN]]
        padded = pad_sequences([sequence], maxlen=RNN_MAX_LEN, padding='post')
        pred = rnn_model.predict(padded, verbose=0)[0][0]
        return ("RNN", float(pred), True)

    def run_cnn():
        sequence = [cnn_char2idx.get(c, 1) for c in str(input.url)[:CNN_MAX_LEN]]
        padded = pad_sequences([sequence], maxlen=CNN_MAX_LEN, padding='post')
        pred = cnn_model.predict(padded, verbose=0)[0][0]
        return ("CNN", float(pred), True)

    def run_nlp():
        if not input.text:
            return None
        sequence = nlp_tokenizer.texts_to_sequences([input.text])
        padded = pad_sequences(sequence, maxlen=NLP_MAX_LEN, padding='post')
        pred = nlp_model.predict(padded, verbose=0)[0][0]
        return ("NLP", float(pred), True)

    def run_dnn():
        if not input.features:
            return None
        f = input.features
        data = np.array([[
            f.UsingIP, f.LongURL, f.ShortURL, f.Symbol,
            f.Redirecting, f.PrefixSuffix, f.SubDomains, f.HTTPS,
            f.DomainRegLen, f.Favicon, f.NonStdPort, f.HTTPSDomainURL,
            f.RequestURL, f.AnchorURL, f.LinksInScriptTags,
            f.ServerFormHandler, f.InfoEmail, f.AbnormalURL,
            f.WebsiteForwarding, f.StatusBarCust, f.DisableRightClick,
            f.UsingPopupWindow, f.IframeRedirection, f.AgeofDomain,
            f.DNSRecording, f.WebsiteTraffic, f.PageRank,
            f.GoogleIndex, f.LinksPointingToPage, f.StatsReport
        ]])
        data_scaled = scaler.transform(data)
        pred = dnn_model.predict(data_scaled, verbose=0)[0][0]
        return ("DNN", float(pred), False)

    # ── Run all 4 models in parallel ─────────────────────────
    futures = [
        loop.run_in_executor(executor, run_rnn),
        loop.run_in_executor(executor, run_cnn),
        loop.run_in_executor(executor, run_nlp),
        loop.run_in_executor(executor, run_dnn),
    ]
    model_outputs = await asyncio.gather(*futures)

    # ── Process results ───────────────────────────────────────
    for item in model_outputs:
        if item is None:
            continue
        model_name, pred, higher_is_phishing = item
        if higher_is_phishing:
            is_phishing = pred > 0.5
            confidence = pred * 100 if is_phishing else (1 - pred) * 100
        else:
            is_phishing = pred < 0.5
            confidence = (1 - pred) * 100 if is_phishing else pred * 100
        if is_phishing:
            phishing_votes += 1
        results.append({
            "model": model_name,
            "prediction": "Phishing" if is_phishing else "Legitimate",
            "confidence": round(confidence, 2)
        })

    total_models = len(results)
    final_verdict = "Phishing" if phishing_votes > total_models / 2 else "Legitimate"
    threat_level = "HIGH" if phishing_votes == total_models else \
                   "MEDIUM" if phishing_votes > 0 else "LOW"

    return {
        "final_verdict": final_verdict,
        "threat_level": threat_level,
        "phishing_votes": f"{phishing_votes}/{total_models}",
        "individual_results": results
    }

# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)