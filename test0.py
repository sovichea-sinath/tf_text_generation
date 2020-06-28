import tensorflow as tf

import numpy as np
import os
import time

path_to_file = 'lotr.txt'

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().lower().decode(encoding='utf-8')

# The unique characters in the file
vocab = sorted(set(text))

# Length of the vocabulary in chars
vocab_size = len(vocab)

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text]) # convert the text to the index

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units, # LSTM is also fine
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

checkpoint_dir = './training_checkpoints'

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()

def startGenerate(model, start_string):
  input_eval1 = [char2idx[s] for s in start_string]
  input_eval1 = tf.expand_dims(input_eval1, 0)

  text_generated = []
  temperature = 1.0

  model.reset_states()
  predictions = model(input_eval1)
  predictions = tf.squeeze(predictions, 0)

  predictions = predictions / temperature
  predicted_ids = tf.random.categorical(predictions, num_samples=1)

  options = []
  for predicted_id in predicted_ids:
    options.append(start_string + idx2char[predicted_id])
  return options

def generate_text(model, start_string):
  model.reset_states()
  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  while (True):
    predictions = model(input_eval)
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)

    # using a categorical distribution to predict the character returned by the model
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    # escape option
    if (idx2char[predicted_id] == ' ' or idx2char[predicted_id] == '\n'):
      break

    # We pass the predicted character as the next input to the model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

option_words = startGenerate(model, start_string=u"lord of the r")

print("option_words: ", option_words)
for option_word in option_words:
  print("final result: ", generate_text(model, start_string=option_word))
