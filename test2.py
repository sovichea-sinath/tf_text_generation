import tensorflow as tf

import numpy as np
import pydash as _
import os
import time

path_to_file = 'allFiles.txt'

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
    tf.keras.layers.Embedding(
      vocab_size, embedding_dim,
      batch_input_shape=[batch_size, None]
    ),
    tf.keras.layers.GRU(
      rnn_units, # LSTM is also fine
      return_sequences=True,
      stateful=True,
      recurrent_initializer='glorot_uniform'
    ),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

model.summary()
def kickStart(model, start_string):
  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  predictions = model(input_eval)
  # remove the batch dimension
  predictions = tf.squeeze(predictions, 0)

  # using a categorical distribution to predict the character returned by the model
  predictions = predictions / temperature
  predicted_ids = tf.random.categorical(predictions, num_samples=1).numpy()
  # returnList = [item for predicted_id in predicted_ids for item in predicted_id]
  returnList = _.flatten(predicted_ids)
  returnList = filter(lambda iterm : ''.join(idx2char[iterm]).isalpha(), returnList)
  return _.uniq(list(map(lambda iterm: start_string + ''.join(idx2char[iterm]), returnList)))

def getPrediction(model, stringList):
  # rerult list
  result = []

  for start_string in stringList:
    for i in range(3):
      result.append(generate_text(model, start_string))
  
  return _.uniq(result)

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  while True:
    predictions = model(input_eval)
    # remove the batch dimension
    predictions = tf.squeeze(predictions, 0)

    # using a categorical distribution to predict the character returned by the model
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
    # break if there is more alphabet left
    if (not ''.join(idx2char[predicted_id]).isalpha()):
      break
    # We pass the predicted character as the next input to the model
    # along with the previous hidden state
    input_eval = tf.expand_dims([predicted_id], 0)

    text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

stringList = kickStart(model, start_string=u"donald t")
print(getPrediction(model, stringList))
