import random 

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import numpy as np 

from sklearn.metrics import accuracy_score, f1_score, classification_report

import gensim 
from gensim.models import KeyedVectors

from seq_tagger_models import LSTMTagger
import preprocessor 


USE_CUDA = False 

def load_init_word_vectors(vectorizer, path_to_wvs=None):
    WVs = KeyedVectors.load_word2vec_format("embeddings/PubMed-w2v.bin", binary=True)

    E = np.zeros((len(vectorizer.str_to_idx), WVs.vector_size))
    WV_matrix = np.matrix([WVs[v] for v in WVs.vocab.keys()])
    mean_vector = np.mean(WV_matrix, axis=0)

    for idx, token in enumerate(vectorizer.idx_to_str):
        if token in WVs:
            E[idx] = WVs[token]
        else:
            E[idx] = mean_vector

    return E 

def get_list_of_ys(y_dicts):
    Y = []
    for y_i in y_dicts.values():
        Y.extend(y_i)
    return set(list(Y))

def to_int_preds(y):
    return [np.argmax(y_i) for y_i in y]


def train(HIDDEN_DIM=32, OUTPUT_SIZE=2, epochs=10, use_val=True):
    v, (Xs, ys) = preprocessor.get_vectorizer(return_data_too=True)

    
    E = load_init_word_vectors(v)
    EMBEDDING_DIM = E.shape[1]

    val_ids = []
    if use_val:
        # (802 because we have 4802 right now in total)
        val_ids = random.sample(Xs.keys(), 802)
        val_X, val_y = [], []
        for val_id in val_ids:
            val_X.append(v.string_to_seq(Xs[val_id]))
            val_y.append(preprocessor._to_torch_var(ys[val_id]))


    train_ids = [id_ for id_ in Xs if not id_ in val_ids]

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(v.str_to_idx), OUTPUT_SIZE)
    if USE_CUDA:
        model.cuda()
    loss_function = nn.CrossEntropyLoss()  
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs):  
        epoch_loss = 0
        # shuffle the examples
        np.random.shuffle(train_ids)
        if epoch > 0:
            for train_id in train_ids:
                x_i, y_i = Xs[train_id], ys[train_id]

                model.zero_grad()
                model.hidden = model.init_hidden()

                x_i_tensor = v.string_to_seq(x_i)
                targets = preprocessor._to_torch_var(y_i)

                #import pdb; pdb.set_trace()
                token_preds = model(x_i_tensor)

             
                loss = loss_function(token_preds, targets)
                epoch_loss += loss 
                loss.backward()
                optimizer.step()

        #import pdb; pdb.set_trace()
        if use_val:
            y_hat_val, y_hat_hard, y_val_flat, val_losses = [], [], [], []
            for i in range(len(val_X)):
                y_hat_i = model(val_X[i])
                y_hat_val.append(y_hat_i)
                val_losses.append(loss_function(y_hat_i, val_y[i]))

                y_hat_hard.extend([int(np.argmax(y_hat_i[x].data)) for x in range(len(y_hat_i))])
                y_val_flat.extend(val_y[i])

            val_loss = sum(val_losses)
            f1 = f1_score(y_val_flat, y_hat_hard)
            #import pdb; pdb.set_trace()
            if epoch == 0:
                print("initial f1 = {}".format(f1))
            else:
                #import pdb; pdb.set_trace()
                print("epoch {}. train loss: {}; val loss: {}; val F1: {:.3f}".format(
                        epoch, epoch_loss.data[0], val_loss.data[0], f1))
        elif epoch > 0:
            print("epoch {}. train loss: {}".format(epoch, epoch_loss.data[0]))

    return model, v


def test(m, v):
    test_docs, test_labels = preprocessor.load_test_data()
    test_X, test_y = preprocessor.dicts_to_X_y(test_docs, test_labels, v)
    
    y_hat, y_hat_hard, y_test_flat = [], [], []
    for i in range(len(test_X)):
        y_hat_i = m(test_X[i])
        y_hat.extend(y_hat_i)
        y_hat_hard.extend([int(np.argmax(y_hat_i[x].data)) for x in range(len(y_hat_i))])
        y_test_flat.extend([int(y_ij) for y_ij in test_y[i]])

    
    f1 = f1_score(y_test_flat, y_hat_hard)

    return f1, y_test_flat, y_hat_hard
   

def test_exp(epochs=10):
    print("!!!WARNING!!! this is using the test data for evaluation -- use sparingly!!")
    m, v = train(epochs=epochs)
    f1, y_test, y_hat = test(m, v)

    print("f1")
    print(f1)


    print ('\n clasification report:\n')

    print(classification_report(y_test, y_hat))





