import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# 1. Load Dataset (Top 10,000 words)
max_features = 10000 
maxlen = 500  # Cut reviews after 500 words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

# 2. Preprocessing: Pad sequences to ensure uniform length
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

# 3. Build the RNN Model
model = Sequential([
    # Embedding layer turns word integers into dense vectors
    Embedding(max_features, 32), 
    # SimpleRNN layer to capture sequential dependencies
    SimpleRNN(32), 
    # Output layer for binary classification (Positive/Negative)
    Dense(1, activation='sigmoid')
])

# 4. Compile and Train
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

# 5. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")
