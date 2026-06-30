import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, GlobalMaxPooling1D, Conv1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Load SMS spam dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['v1', 'v2']].copy()
df.columns = ['label', 'text']
df = df.dropna()
df['text'] = df['text'].astype(str)

# Map labels
label_map = {'ham': 0, 'spam': 1}
df['label'] = df['label'].map(label_map)

# Tokenize
MAX_WORDS = 10000
MAX_LEN = 100
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
y = df['label'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Build model
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=128),
    Conv1D(64, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                    validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Save
model.save('phishguard_nlp_model.h5')
with open('nlp_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('nlp_label_map.pkl', 'wb') as f:
    pickle.dump(label_map, f)

print("✅ NLP Model saved!")
print("Accuracy:", round(history.history['val_accuracy'][-1] * 100, 2), "%")