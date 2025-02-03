from typing import Literal


device = "mps"
d_model = 768

# Word2Vec
window_size = 8
method: Literal["cbow", "skipgram"] = "skipgram"
lr_word2vec = 5e-04
num_epochs_word2vec = 10
batch_size_word2vec = 256
num_workers = 4

# GRU
hidden_size = 768
num_classes = 4
lr = 5e-05
num_epochs = 15
batch_size = 16