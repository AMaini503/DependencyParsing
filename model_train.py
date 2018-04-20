import tensorflow as tf
import numpy as np
from aux import replaceUsingIndex, getIndexFromFile, LoadNewTrainFile


# create indices from vocab files
index_of_words = getIndexFromFile(filename = 'data/vocabs.word')
index_of_labels = getIndexFromFile(filename = 'data/vocabs.labels')
index_of_pos = getIndexFromFile(filename = 'data/vocabs.pos')
index_of_actions = getIndexFromFile(filename = 'data/vocabs.actions')


n_words = len(index_of_words)
n_tags = len(index_of_pos)
n_labels = len(index_of_labels)
n_actions = len(index_of_actions)


new_train_file = 'data/train_with_indices.data'

# Create temporary training file
replaceUsingIndex(oldfilename = 'data/train.data', newfilename = new_train_file, 
                  indices = [index_of_words, index_of_pos, index_of_labels, index_of_actions])



# Load feature matrix and target labels from the new training file
train_data, train_labels = LoadNewTrainFile(filename = 'data/train_with_indices.data')

from keras.models import Model
from keras.layers import Dense, Input, Embedding, Reshape, Concatenate, Lambda
import keras


# Dimension of word embedding
dw = 64

# Dimension of tag embeddings
dt = 32

# Dimension of dependency label embeddings
dl = 32




X = Input(shape = (52,))

words = Lambda(function = lambda x: x[:, 0:20], output_shape = (256, 20))(X)
tags = Lambda(function = lambda x: x[:, 20:20 + 20], output_shape = (256, 20))(X)
labels = Lambda(function = lambda x: x[:, 40: 40 + 12], output_shape = (256, 12))(X)

embedding_words = Embedding(
    input_dim = n_words,
    output_dim = 64,
    input_length = 20,
    embeddings_initializer = 'normal'
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
embeddings = Concatenate()([embedding_words, embedding_tags, embedding_labels])

h1 = Dense(units = 200, activation = 'relu')(embeddings)
h2 = Dense(units = 200, activation = 'relu')(h1)

q = Dense(units = 93, activation = 'softmax')(h2)


# In[13]:


model = Model(inputs = [X], outputs = [q])
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')


# In[14]:


train_labels_oh = keras.utils.to_categorical(train_labels, num_classes = n_actions)
train_labels_oh.shape


# In[15]:


model.fit(train_data, train_labels_oh, epochs = 7, batch_size = 256)


# In[16]:


model.save(filepath = 'saved_models/model1.h5')

