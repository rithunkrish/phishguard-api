from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import bisect
import os
from urllib.parse import urlparse

app = FastAPI(title="PhishGuard API — Quad Model Protection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Load DNN
dnn_model = tf.keras.models.load_model("phishing_model_local.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("✅ DNN Model loaded!")

# Load NLP
nlp_model = tf.keras.models.load_model("phishguard_nlp_model.h5")
with open("nlp_tokenizer.pkl", "rb") as f:
    nlp_tokenizer = pickle.load(f)
with open("nlp_label_map.pkl", "rb") as f:
    nlp_label_map = pickle.load(f)
NLP_MAX_LEN = 100
print("✅ NLP Model loaded!")

# Load RNN
rnn_model = tf.keras.models.load_model("phishguard_rnn_model.h5")
with open("rnn_char2idx.pkl", "rb") as f:
    rnn_char2idx = pickle.load(f)
RNN_MAX_LEN = 200
print("✅ RNN Model loaded!")

# Load CNN
cnn_model = tf.keras.models.load_model("phishguard_cnn_model.h5")
with open("cnn_char2idx.pkl", "rb") as f:
    cnn_char2idx = pickle.load(f)
CNN_MAX_LEN = 200
print("✅ CNN Model loaded!")

# ============================================================
# Tranco website-traffic lookup
# ------------------------------------------------------------
# tranco_lookup.txt is a pre-sorted (by domain) "domain,rank" file
# built once via build_tranco_lookup.py. We load it into two parallel
# lists at startup so we can binary-search it instantly per request,
# instead of scanning a million lines on every API call.
# ============================================================
TRANCO_FILE = "tranco_lookup.txt"
tranco_domains = []   # sorted list of domain strings
tranco_ranks = []     # parallel list of int ranks

if os.path.exists(TRANCO_FILE):
    with open(TRANCO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            domain, rank = line.rsplit(",", 1)
            tranco_domains.append(domain)
            tranco_ranks.append(int(rank))
    print(f"✅ Tranco lookup loaded! ({len(tranco_domains):,} domains)")
else:
    print("⚠️  tranco_lookup.txt not found — WebsiteTraffic will default to neutral (0) for all sites.")
    print("    Run build_tranco_lookup.py to generate it (see project notes).")


def get_tranco_rank(domain: str):
    """Binary-search the sorted Tranco list. Returns rank (int) or None if not found."""
    if not tranco_domains:
        return None

    # Tranco stores bare registrable domains (e.g. "google.com"), not "www.google.com",
    # so strip a leading "www." before searching — otherwise every www-prefixed site
    # (which is most of them) would incorrectly come back as "not found".
    if domain.startswith("www."):
        domain = domain[4:]

    idx = bisect.bisect_left(tranco_domains, domain)
    if idx < len(tranco_domains) and tranco_domains[idx] == domain:
        return tranco_ranks[idx]
    return None


def get_website_traffic_score(domain: str) -> int:
    """
    Maps a domain to the dataset's WebsiteTraffic convention:
      1  = well-trafficked (top 100,000 on Tranco)
      0  = ranked, but lower traffic
     -1  = not found in Tranco's top 1M at all (unranked / very obscure)
    """
    rank = get_tranco_rank(domain)
    if rank is None:
        return -1
    if rank <= 100_000:
        return 1
    return 0


# ============================================================
# URL-string-only DNN features (category 1)
# ------------------------------------------------------------
# These need only the URL itself — no page content, no external data.
# Conventions follow the original UCI "Phishing Websites" dataset paper.
# ============================================================
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "shorte.st", "rebrand.ly", "tiny.cc", "cutt.ly", "lnkd.in"
}


def get_url_based_features(url: str) -> dict:
    """
    Computes the subset of the 30 DNN features that can be derived purely
    from the URL string itself (no DOM access, no external lookups).
    Everything else in the 30-feature set is filled in separately
    (Tranco for WebsiteTraffic, neutral defaults for the rest, DOM-based
    features to be added in a later step via content.js).
    """
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url  # urlparse needs a scheme to extract hostname reliably

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    # UsingIP: hostname is a raw IPv4 address -> suspicious
    is_ip = bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname))
    using_ip = -1 if is_ip else 1

    # LongURL: dataset thresholds (paper convention)
    url_len = len(url)
    if url_len < 54:
        long_url = 1
    elif url_len <= 75:
        long_url = 0
    else:
        long_url = -1

    # ShortURL: known shortener service
    short_url = -1 if hostname in SHORTENER_DOMAINS else 1

    # Symbol@: literal "@" anywhere in the URL (used to obscure real destination)
    symbol_at = -1 if "@" in url else 1

    # Redirecting//: a second "//" appearing after the protocol (position > 7)
    last_double_slash = url.rfind("//")
    redirecting = -1 if last_double_slash > 7 else 1

    # PrefixSuffix-: hyphen in the domain itself (not path) — common in fake-brand domains
    prefix_suffix = -1 if "-" in hostname else 1

    # SubDomains: number of dots in hostname
    dot_count = hostname.count(".")
    if dot_count <= 1:
        sub_domains = 1
    elif dot_count == 2:
        sub_domains = 0
    else:
        sub_domains = -1

    # HTTPS
    https = 1 if parsed.scheme == "https" else -1

    return {
        "UsingIP": using_ip,
        "LongURL": long_url,
        "ShortURL": short_url,
        "Symbol": symbol_at,
        "Redirecting": redirecting,
        "PrefixSuffix": prefix_suffix,
        "SubDomains": sub_domains,
        "HTTPS": https,
        "hostname": hostname,  # returned for convenience (e.g. Tranco lookup), strip before sending to model
    }


# Features we don't compute yet — neutral default for each.
# Confirmed via feature-importance analysis that these 7 barely move the
# DNN's prediction (each under 1.6% importance), so a fixed neutral value
# is a reasonable placeholder until/unless real data sources are wired up.
NEUTRAL_DEFAULT_FEATURES = {
    "DomainRegLen": 0,
    "AgeofDomain": 0,
    "DNSRecording": 0,
    "PageRank": 0,
    "GoogleIndex": 0,
    "LinksPointingToPage": 0,
    "StatsReport": 0,
}

# DOM-based features (category 2) — not wired up yet either.
# Defaulting to neutral for now; these will be computed by content.js
# in a future step (Favicon, RequestURL, AnchorURL, etc.)
DOM_PLACEHOLDER_FEATURES = {
    "Favicon": 0,
    "NonStdPort": 0,
    "HTTPSDomainURL": 0,
    "RequestURL": 0,
    "AnchorURL": 0,
    "LinksInScriptTags": 0,
    "ServerFormHandler": 0,
    "InfoEmail": 0,
    "AbnormalURL": 0,
    "WebsiteForwarding": 0,
    "StatusBarCust": 0,
    "DisableRightClick": 0,
    "UsingPopupWindow": 0,
    "IframeRedirection": 0,
}


def build_full_dnn_features(url: str) -> dict:
    """
    Assembles all 30 DNN features for a given URL:
      - real values for URL-string features (category 1)
      - real value for WebsiteTraffic via Tranco
      - neutral defaults for everything else (category 2 DOM features,
        and the 7 confirmed-low-importance external-data features)
    """
    url_features = get_url_based_features(url)
    hostname = url_features.pop("hostname")

    features = {
        **url_features,
        **DOM_PLACEHOLDER_FEATURES,
        **NEUTRAL_DEFAULT_FEATURES,
        "WebsiteTraffic": get_website_traffic_score(hostname),
    }
    return features


print("\n🛡️ PhishGuard API — All 4 models ready!")


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
        "endpoints": ["/predict/dnn", "/predict/nlp", "/predict/rnn", "/predict/cnn", "/predict/combined", "/extract-dnn-features"]
    }


@app.post("/extract-dnn-features")
def extract_dnn_features(input: URLInput):
    """
    Given just a URL, computes the full 30-feature DNN input:
    real values where we can derive them (URL string + Tranco traffic rank),
    neutral defaults everywhere else. Lets the extension call /predict/dnn
    (or get DNN included in /predict/combined) without building its own
    feature-extraction logic client-side.
    """
    return build_full_dnn_features(input.url)


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


@app.post("/predict/combined")
def predict_combined(input: CombinedInput):
    results = []
    phishing_votes = 0

    # RNN
    rnn_sequence = [rnn_char2idx.get(c, 1) for c in str(input.url)[:RNN_MAX_LEN]]
    rnn_padded = pad_sequences([rnn_sequence], maxlen=RNN_MAX_LEN, padding='post')
    rnn_pred = rnn_model.predict(rnn_padded, verbose=0)[0][0]
    rnn_is_phishing = rnn_pred > 0.5
    rnn_confidence = rnn_pred * 100 if rnn_is_phishing else (1 - rnn_pred) * 100
    if rnn_is_phishing: phishing_votes += 1
    results.append({"model": "RNN", "prediction": "Phishing" if rnn_is_phishing else "Legitimate", "confidence": round(float(rnn_confidence), 2)})

    # CNN
    cnn_sequence = [cnn_char2idx.get(c, 1) for c in str(input.url)[:CNN_MAX_LEN]]
    cnn_padded = pad_sequences([cnn_sequence], maxlen=CNN_MAX_LEN, padding='post')
    cnn_pred = cnn_model.predict(cnn_padded, verbose=0)[0][0]
    cnn_is_phishing = cnn_pred > 0.5
    cnn_confidence = cnn_pred * 100 if cnn_is_phishing else (1 - cnn_pred) * 100
    if cnn_is_phishing: phishing_votes += 1
    results.append({"model": "CNN", "prediction": "Phishing" if cnn_is_phishing else "Legitimate", "confidence": round(float(cnn_confidence), 2)})

    # NLP
    if input.text:
        nlp_sequence = nlp_tokenizer.texts_to_sequences([input.text])
        nlp_padded = pad_sequences(nlp_sequence, maxlen=NLP_MAX_LEN, padding='post')
        nlp_pred = nlp_model.predict(nlp_padded, verbose=0)[0][0]
        nlp_is_phishing = nlp_pred > 0.5
        nlp_confidence = nlp_pred * 100 if nlp_is_phishing else (1 - nlp_pred) * 100
        if nlp_is_phishing: phishing_votes += 1
        results.append({"model": "NLP", "prediction": "Phishing" if nlp_is_phishing else "Legitimate", "confidence": round(float(nlp_confidence), 2)})

    # DNN — use explicitly-provided features if given, otherwise auto-derive
    # them from the URL (real URL-string + Tranco values, neutral defaults
    # for the rest) so DNN always participates in the combined verdict.
    if input.features:
        f = input.features
        feature_values = [
            f.UsingIP, f.LongURL, f.ShortURL, f.Symbol,
            f.Redirecting, f.PrefixSuffix, f.SubDomains, f.HTTPS,
            f.DomainRegLen, f.Favicon, f.NonStdPort, f.HTTPSDomainURL,
            f.RequestURL, f.AnchorURL, f.LinksInScriptTags,
            f.ServerFormHandler, f.InfoEmail, f.AbnormalURL,
            f.WebsiteForwarding, f.StatusBarCust, f.DisableRightClick,
            f.UsingPopupWindow, f.IframeRedirection, f.AgeofDomain,
            f.DNSRecording, f.WebsiteTraffic, f.PageRank,
            f.GoogleIndex, f.LinksPointingToPage, f.StatsReport
        ]
    else:
        auto_features = build_full_dnn_features(input.url)
        feature_values = [
            auto_features["UsingIP"], auto_features["LongURL"], auto_features["ShortURL"], auto_features["Symbol"],
            auto_features["Redirecting"], auto_features["PrefixSuffix"], auto_features["SubDomains"], auto_features["HTTPS"],
            auto_features["DomainRegLen"], auto_features["Favicon"], auto_features["NonStdPort"], auto_features["HTTPSDomainURL"],
            auto_features["RequestURL"], auto_features["AnchorURL"], auto_features["LinksInScriptTags"],
            auto_features["ServerFormHandler"], auto_features["InfoEmail"], auto_features["AbnormalURL"],
            auto_features["WebsiteForwarding"], auto_features["StatusBarCust"], auto_features["DisableRightClick"],
            auto_features["UsingPopupWindow"], auto_features["IframeRedirection"], auto_features["AgeofDomain"],
            auto_features["DNSRecording"], auto_features["WebsiteTraffic"], auto_features["PageRank"],
            auto_features["GoogleIndex"], auto_features["LinksPointingToPage"], auto_features["StatsReport"]
        ]

    data = np.array([feature_values])
    data_scaled = scaler.transform(data)
    dnn_pred = dnn_model.predict(data_scaled, verbose=0)[0][0]
    dnn_is_phishing = dnn_pred < 0.5
    dnn_confidence = (1 - dnn_pred) * 100 if dnn_is_phishing else dnn_pred * 100
    if dnn_is_phishing: phishing_votes += 1
    results.append({"model": "DNN", "prediction": "Phishing" if dnn_is_phishing else "Legitimate", "confidence": round(float(dnn_confidence), 2)})

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
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)