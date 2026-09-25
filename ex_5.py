!pip -q install tensorflow
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
print("Name: Balaji.S")
print("Reg.No: 814724243022")
vocab_size = 10000
max_length = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)
model = Sequential([
Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
LSTM(64),
Dense(1, activation="sigmoid")
])
model.compile(
optimizer="adam",
loss="binary_crossentropy",
metrics=["accuracy"]
model.fit(
x_train[:10000],
y_train[:10000],
epochs=3,
batch_size=64,
validation_split=0.2
loss, accuracy = model.evaluate(x_test[:5000], y_test[:5000])
print("Test Accuracy:", accuracy)
prediction = model.predict(x_test[:1])
if prediction[0][0] >= 0.5:
print("Sentiment: Positive")
else:
print("Sentiment: Negative")
