import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Embedding, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Load dataset
df = pd.read_csv("data.csv")
df['label'] = df['label'].map({'good': 0, 'bad': 1})
df = df.dropna()
print("✅ Dataset loaded:", df.shape)

# Character level encoding
all_chars = set(''.join(df['url'].astype(str)))
char2idx = {char: idx+2 for idx, char in enumerate(sorted(all_chars))}
char2idx['<PAD>'] = 0
char2idx['<UNK>'] = 1

MAX_LEN = 200
VOCAB_SIZE = len(char2idx)

def url_to_sequence(url):
    return [char2idx.get(c, 1) for c in str(url)[:MAX_LEN]]

print("Converting URLs to sequences...")
sequences = df['url'].apply(url_to_sequence).tolist()
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
y = df['label'].values

# Sample and split
idx = np.random.choice(len(X), 100000, replace=False)
X_sample = X[idx]
y_sample = y[idx]

X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample)

print("✅ Training samples:", X_train.shape)

# Build CNN model
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=64),
    Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    Conv1D(filters=32, kernel_size=7, activation='relu', padding='same'),
    GlobalMaxPooling1D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=15, batch_size=64,
                    validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {round(test_accuracy * 100, 2)}%")

y_pred = (model.predict(X_test) > 0.5).astype(int)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

# Save
model.save('phishguard_cnn_model.h5')
with open('cnn_char2idx.pkl', 'wb') as f:
    pickle.dump(char2idx, f)

print("\n✅ CNN Model saved as phishguard_cnn_model.h5")
print("✅ Vocabulary saved as cnn_char2idx.pkl")
print(f"\n🎉 Final Accuracy: {round(history.history['val_accuracy'][-1] * 100, 2)}%")