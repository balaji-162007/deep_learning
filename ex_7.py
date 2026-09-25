!wget -q https://storage.googleapis.com/download.tensorflow.org/data/fra-eng.zip
!unzip -o -q fra-eng.zip
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
print("Name: Balaji.S")
print("Reg.No: 814724243022")
batch_size = 64
epochs = 100
latent_dim = 256
num_samples = 10000
data_path = "fra.txt"
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
lines = f.read().split("\n")
for line in lines[:min(num_samples, len(lines)-1)]:
parts = line.split("\t")
if len(parts) < 2:
continue
input_text = parts[0]
target_text = "\t" + parts[1] + "\n"
input_texts.append(input_text)

target_texts.append(target_text)
input_characters.update(input_text)
target_characters.update(target_text)
input_characters = sorted(input_characters)
target_characters = sorted(target_characters)
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max(len(txt) for txt in input_texts)
max_decoder_seq_length = max(len(txt) for txt in target_texts)
print("Number of samples:", len(input_texts))
print("Input tokens:", num_encoder_tokens)
print("Output tokens:", num_decoder_tokens)
input_token_index = {char: i for i, char in enumerate(input_characters)}
target_token_index = {char: i for i, char in enumerate(target_characters)}

encoder_input_data = np.zeros(
(len(input_texts), max_encoder_seq_length, num_encoder_tokens),
dtype="float32"
)
decoder_input_data = np.zeros(
(len(input_texts), max_decoder_seq_length, num_decoder_tokens),
dtype="float32"
)
decoder_target_data = np.zeros(
(len(input_texts), max_decoder_seq_length, num_decoder_tokens),
dtype="float32"

)
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
for t, char in enumerate(input_text):
encoder_input_data[i, t, input_token_index[char]] = 1.0
encoder_input_data[i, t+1:, input_token_index[" "]] = 1.0
for t, char in enumerate(target_text):
decoder_input_data[i, t, target_token_index[char]] = 1.0
if t > 0:
decoder_target_data[i, t-1, target_token_index[char]] = 1.0
decoder_input_data[i, t+1:, target_token_index[" "]] = 1.0
decoder_target_data[i, t:, target_token_index[" "]] = 1.0
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(
optimizer="rmsprop",
loss="categorical_crossentropy",
metrics=["accuracy"]
)

model.fit(
[encoder_input_data, decoder_input_data],
decoder_target_data,
batch_size=batch_size,
epochs=epochs,
validation_split=0.2
)
model.save("eng2french.h5")
print("Model saved successfully!")
