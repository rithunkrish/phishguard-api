import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100

# Load model and tokenizer
model = tf.keras.models.load_model('phishguard_nlp_model.h5')
with open('nlp_tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('nlp_label_map.pkl', 'rb') as f:
    label_map = pickle.load(f)

print("✅ NLP Model loaded successfully!")
print("Label map:", label_map)

def predict_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
    prediction = model.predict(padded, verbose=0)[0][0]
    result = "Phishing" if prediction > 0.5 else "Legitimate"
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    print(f"\nText: {text[:80]}")
    print(f"Prediction: {result}")
    print(f"Confidence: {round(confidence, 2)}%")

# Test cases
predict_text("URGENT! Your bank account has been suspended. Click here to verify now!")
predict_text("FREE prize winner! Claim your reward at http://win-prize-now.com")
predict_text("Hi John, are you coming to the party tonight?")
predict_text("Your package has been delivered successfully.")