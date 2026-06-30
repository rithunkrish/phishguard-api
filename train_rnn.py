import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ── 1. Load Dataset ──────────────────────────────────────────
df = pd.read_csv("data.csv")
df['label'] = df['label'].map({'good': 0, 'bad': 1})
df = df.dropna()
print("✅ Dataset loaded:", df.shape)
print("Label distribution:")
print(df['label'].value_counts())

# ── 2. Character Level Tokenization ──────────────────────────
all_chars = set(''.join(df['url'].astype(str)))
char2idx = {char: idx+2 for idx, char in enumerate(sorted(all_chars))}
char2idx['<PAD>'] = 0
char2idx['<UNK>'] = 1

MAX_LEN = 200
VOCAB_SIZE = len(char2idx)
print(f"\n✅ Vocabulary size: {VOCAB_SIZE} characters")

def url_to_sequence(url):
    return [char2idx.get(c, 1) for c in str(url)[:MAX_LEN]]

print("Converting URLs to sequences...")
sequences = df['url'].apply(url_to_sequence).tolist()
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
y = df['label'].values

# ── 3. Sample and Split ───────────────────────────────────────
idx = np.random.choice(len(X), 100000, replace=False)
X_sample = X[idx]
y_sample = y[idx]

X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample)

print(f"\n✅ Training samples: {X_train.shape}")
print(f"✅ Testing samples:  {X_test.shape}")

# ── 4. Build Model ────────────────────────────────────────────
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=64),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ── 5. Train Model ────────────────────────────────────────────
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

print("\n🚀 Training started...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# ── 6. Print Training Summary ─────────────────────────────────
print("\n" + "="*50)
print("📊 TRAINING SUMMARY")
print("="*50)
for epoch in range(len(history.history['accuracy'])):
    train_acc  = round(history.history['accuracy'][epoch] * 100, 2)
    val_acc    = round(history.history['val_accuracy'][epoch] * 100, 2)
    train_loss = round(history.history['loss'][epoch], 4)
    val_loss   = round(history.history['val_loss'][epoch], 4)
    print(f"Epoch {epoch+1:02d} | "
          f"Train Acc: {train_acc}% | "
          f"Val Acc: {val_acc}% | "
          f"Train Loss: {train_loss} | "
          f"Val Loss: {val_loss}")

print("="*50)
print(f"✅ Best Training Accuracy:   {round(max(history.history['accuracy']) * 100, 2)}%")
print(f"✅ Best Validation Accuracy: {round(max(history.history['val_accuracy']) * 100, 2)}%")
print(f"✅ Best Training Loss:       {round(min(history.history['loss']), 4)}")
print(f"✅ Best Validation Loss:     {round(min(history.history['val_loss']), 4)}")
print("="*50)

# ── 7. Evaluate on Test Set ───────────────────────────────────
print("\n📊 EVALUATING ON TEST SET...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {round(test_accuracy * 100, 2)}%")
print(f"✅ Test Loss:     {round(test_loss, 4)}")

y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\n📊 CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred,
      target_names=['Legitimate', 'Phishing']))

# ── 8. Plot Training Curves ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('PhishGuard RNN Model — Training Metrics', fontsize=14, fontweight='bold')

# Accuracy plot
axes[0].plot(history.history['accuracy'], 
             label='Training Accuracy', color='#00d4ff', linewidth=2)
axes[0].plot(history.history['val_accuracy'], 
             label='Validation Accuracy', color='#7b2ff7', linewidth=2)
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], 
             label='Training Loss', color='#00ff88', linewidth=2)
axes[1].plot(history.history['val_loss'], 
             label='Validation Loss', color='#ff4444', linewidth=2)
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rnn_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Training curves saved as rnn_training_curves.png")

# ── 9. Confusion Matrix ───────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'])
plt.title('PhishGuard RNN — Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('rnn_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrix saved as rnn_confusion_matrix.png")

# ── 10. Test with Sample URLs ─────────────────────────────────
print("\n📊 SAMPLE PREDICTIONS:")
print("="*50)

def predict_url(url):
    sequence = [char2idx.get(c, 1) for c in str(url)[:MAX_LEN]]
    padded = pad_sequences([sequence], maxlen=MAX_LEN, padding='post')
    prediction = model.predict(padded, verbose=0)[0][0]
    result = "🚨 Phishing" if prediction > 0.5 else "✅ Legitimate"
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    print(f"URL: {url}")
    print(f"Prediction: {result} | Confidence: {round(confidence, 2)}%")
    print("-"*50)

predict_url("paypal-secure-login-verify.com/account")
predict_url("192.168.1.1/login/bank/verify")
predict_url("secure-paypal-update.net/signin")
predict_url("google.com")
predict_url("amazon.com/orders")
predict_url("github.com/flutter")

# ── 11. Save Model ────────────────────────────────────────────
model.save('phishguard_rnn_model.h5')
with open('rnn_char2idx.pkl', 'wb') as f:
    pickle.dump(char2idx, f)

print("\n✅ Model saved as phishguard_rnn_model.h5")
print("✅ Vocabulary saved as rnn_char2idx.pkl")
print("\n🎉 RNN Training Complete!")