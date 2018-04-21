import tensorflow as tf
import numpy as np
from aux import replaceUsingIndex, getIndexFromFile, LoadNewTrainFile, LoadTestFile
import sys


# create indices from vocab files
index_of_words = getIndexFromFile(filename = 'data/vocabs.word')
index_of_labels = getIndexFromFile(filename = 'data/vocabs.labels')
index_of_pos = getIndexFromFile(filename = 'data/vocabs.pos')
index_of_actions = getIndexFromFile(filename = 'data/vocabs.actions')


n_words = len(index_of_words)
n_tags = len(index_of_pos)
n_labels = len(index_of_labels)
n_actions = len(index_of_actions)
print(n_words, n_tags, n_labels)

new_train_file = 'data/train_with_indices.data'
#new_test_file = 'data/dev_with_indices.data'

# Create temporary training file
replaceUsingIndex(oldfilename = 'data/train.data', newfilename = new_train_file, 
                  indices = [index_of_words, index_of_pos, index_of_labels, index_of_actions])

# Create temporary test file
#replaceUsingIndex(oldfilename = 'data/dev.data', newfilename = new_test_file, 
#                  indices = [index_of_words, index_of_pos, index_of_labels, index_of_actions])



# Load feature matrix and target labels from the new training file
train_data, train_labels = LoadNewTrainFile(filename = 'data/train_with_indices.data')
# test_data, test_labels = LoadTestFile(filename = 'data/dev_with_indices.data')

print(train_data.shape, train_labels.shape)
from keras.models import Model
from keras.layers import Dense, Input, Embedding, Reshape, Concatenate, Lambda, LeakyReLU, Dropout
import keras


# Dimension of word embedding
dw = 64

# Dimension of tag embeddings
dt = 32

# Dimension of dependency label embeddings
dl = 32

def output_shape_words(input_shape):
    assert(len(list(input_shape)) == 2)
    assert(input_shape[1] == 52)
    return (input_shape[0], 20)

def output_shape_tags(input_shape):
    assert(len(list(input_shape)) == 2)
    assert(input_shape[1] == 52)
    return (input_shape[0], 20)

def output_shape_labels(input_shape):
    assert(len(list(input_shape)) == 2)
    assert(input_shape[1] == 52)
    return (input_shape[0], 12)


X = Input(shape = (52, ))


words = Lambda(function = lambda x: x[:, 0: 20], output_shape = output_shape_words)(X)
tags = Lambda(function = lambda x: x[:, 20: 20 + 20], output_shape = output_shape_tags)(X)
labels = Lambda(function = lambda x: x[:, 40: 40 + 41], output_shape = output_shape_labels)(X)

embedding_words = Embedding(
    input_dim = n_words,
    output_dim = 64,
    input_length = 20,
)(words)

embedding_words = Reshape(target_shape = (20 * 64, ))(embedding_words)

embedding_tags = Embedding(
    input_dim = n_tags,
    output_dim = 32,
    input_length = 20
)(tags)

embedding_tags = Reshape(target_shape = (32 * 20,) )(embedding_tags)

embedding_labels = Embedding(
    input_dim = n_labels,
    output_dim = 32,
    input_length = 12
)(labels)

embedding_labels = Reshape(target_shape = (32 * 12, ))(embedding_labels)


# concatenate the embeddings
embeddings = Concatenate(axis = 1)([embedding_words, embedding_tags, embedding_labels])

h1 = Dense(units = 500)(embeddings)
h1_lrelu = LeakyReLU()(h1)
h1_do = Dropout(0.2)(h1_lrelu)
h2 = Dense(units = 500)(h1_do)
h2_lrelu = LeakyReLU()(h2)
h2_do = Dropout(0.2)(h2_lrelu)

q = Dense(units = 93, activation = 'softmax')(h2_do)

model = Model(inputs = [X], outputs = [q])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
#
print(model.summary())
model.fit(train_data, train_labels, epochs = 15, batch_size = 1000)


# In[16]:


model.save(filepath = 'saved_models/model8.h5')

