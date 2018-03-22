import keras
import numpy as np

from keras.preprocessing.text import Tokenizer, hashing_trick
from keras.layers.embeddings import Embedding

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)

# This turns strings into lists of integer indices.
sequences = tokenizer.texts_to_sequences(samples)

# One hot encoding
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

# Word index
word_index = tokenizer.word_index

# Word-level one-hot encoding with hashing trick
sentence1 = hashing_trick(samples[0], 256)
sentence2 = hashing_trick(samples[1], 256)
