'''
#Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.


You can try some other texts too:


What about Tolstoys Anna Karenina:
https://raw.githubusercontent.com/udacity/deep-learning/master/tensorboard/anna.txt

Or some Nietzsche:
https://s3.amazonaws.com/text-datasets/nietzsche.txt

Germany Wikipedia Articles:
https://www2.htw-dresden.de/~guhr/dist/wiki.txt

Shakesspears Sonnets:
https://raw.githubusercontent.com/vivshaw/shakespeare-LSTM/master/sonnets.txt
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback, TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, CuDNNLSTM, CuDNNGRU, Dropout
from keras.optimizers import RMSprop, SGD, Nadam
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
from datetime import datetime
import re

path = get_file(
    'shakespear.txt',
    origin='https://cs.stanford.edu/people/karpathy/char-rnn/shakespear.txt')


with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

# build lookup table
chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
# How does the network react when you change the sequence length or stepsize
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM layer
# experiment: 
# - add some more neurons
# - add some more layers
# - add dropout 
# - try out GRU's 

print('Build model...')
model = Sequential()
model.add(CuDNNLSTM(128,input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

rms = RMSprop(lr=0.01) 
# try some other optimizers
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy', optimizer=rms)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    # Read more about this softmax with temperature here: 
    # Distilling the Knowledge in a Neural Network (Geoffrey Hinton, Oriol Vinyals, Jeff Dean)
    # https://arxiv.org/abs/1503.02531
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.8, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)
        sys.stdout.write("----- result ------")
        for i in range(300):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

# print some text with the current model
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# train the model
model.fit(x, y,
          batch_size=128,
          epochs=90,
          callbacks=[print_callback])

# save the model
model.save("shakespear-rnn")          