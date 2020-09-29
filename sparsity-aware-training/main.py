#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

sys.path.append("DeepCTR-Torch")

# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat
from deepctr_torch.models import *

import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


# In[2]:


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=('deepfm',
                                        'wdl','nfm','din', 'dcn'),
                    default='nfm')

parser.add_argument("--dense-opt", choices=('adam',
                                            'sgd', 'adagrad',), default='sgd')
parser.add_argument("--sparse-opt", choices=('adam',
                                             'sgd', 'adagrad', 'radagrad', 'None'), default='sgd')

parser.add_argument("--dense-lr", type=float, default=0.01)
parser.add_argument("--sparse-lr", type=float, default=0.001)
parser.add_argument("--dataset", choices=('criteo','avazu', 'movielen'), default='avazu')
parser.add_argument("--debug", action = 'store_true', default=False)
args = parser.parse_args()

# In[3]:


data_prefix = "/data/project/deep-ctr-torch/"

if args.dataset == "avazu":
    data_prefix = data_prefix + "avazu-ctr-prediction/"
    if args.debug:
        df = pd.read_csv(data_prefix + "train-mini")
    else:
        df = pd.read_csv(data_prefix + "train")
    data = df
    sparse_features = ['id', 'C1', 'banner_pos', 'site_id', 'site_domain',
                       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
                       'device_ip', 'device_model', 'device_type', 'device_conn_type', ] \
                      + ['C' + str(i) for i in range(14, 22)]
    dense_features = []

    target = ['click']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features

    mms = MinMaxScaler(feature_range=(0, 1))
    if dense_features != []:
        data[dense_features] = mms.fit_transform(data[dense_features])

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique() + 10, embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                                                                            for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for modelf

    train, test = train_test_split(data, test_size=0.1)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

elif args.dataset == "criteo":
    # Data processing code adapted from https://github.com/facebookresearch/dlrm
    # Follow steps in https://github.com/ylongqi/dlrm/blob/master/data_utils.py to generate kaggle_processed.npz
    # Or using `./download_dataset.sh criteo` command to download the processed data.
    import os
    datapath = data_prefix + "criteo/"
    if args.debug:
        datapath = datapath + 'kaggle_processed_tiny.npz'
    else:
        datapath = datapath + 'kaggle_processed.npz'

    import numpy as np
    with np.load(datapath) as data:
        X_int = data["X_int"]
        X_cat = data["X_cat"]
        y = data["y"]
        counts = data["counts"]

    raw_data = dict()

    raw_data['counts'] = counts
    X_cat = X_cat.astype(np.int32)
    X_int = np.log(X_int + 1).astype(np.float32)

    int_df = pd.DataFrame(X_int,columns=['I' + str(i) for i in range(1, 14)])
    cat_df = pd.DataFrame(X_cat,columns=['C' + str(i) for i in range(1, 27)])
    target_df = pd.DataFrame(y,columns=['label'])
    df = pd.concat([int_df,cat_df,target_df], axis=1)
    data = df
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.1)

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

elif args.dataset == "movielen":

    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    data_prefix = data_prefix + "ml_25m/"
    if args.debug:
        datapath = data_prefix + "ml_25m_test-mini.csv"
    else:
        datapath = data_prefix + "ml_25m_test.csv"
    data = pd.read_csv(datapath)
    sparse_features = ["movieId", "userId",'imdbId','tmdbId']
    target = ['rating']

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # preprocess the sequence feature

    key2index = {}
    genres_list = list(map(split, data['genres'].values))
    genres_length = np.array(list(map(len, genres_list)))
    max_len = max(genres_length)
    # Notice : padding=`post`
    genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )

    # 2.count #unique features for each sparse field and generate feature config for sequence feature

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=4)
                              for feat in sparse_features]

    varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
        key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
                                               length_name=None)]  # Notice : value 0 is for padding for sequence input feature

    linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
#     model_input = {name: data[name] for name in sparse_features}  #
#     model_input["genres"] = genres_list

    data['genres'] = genres_list
    train, test = train_test_split(data, test_size=0.1)
    feature_names = ["movieId", "userId",'imdbId','tmdbId', 'genres']
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}


#     history = model.fit(model_input, data[target].values,
#                         batch_size=256, epochs=1, verbose=1, validation_split=0.1, verbose_steps=1000)




# In[4]:


# 4.Define Model,train,predict and evaluate
model_name = args.model
optimizer_dense = args.dense_opt
optimizer_sparse = args.sparse_opt
optimizer_dense_lr = args.dense_lr
optimizer_sparse_lr = args.sparse_lr

if optimizer_sparse == 'None':
    optimizer_sparse = None


print("=====", model_name,optimizer_dense,optimizer_dense_lr,optimizer_sparse,optimizer_sparse_lr, "=====")
device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

model = None
if model_name == "deepfm":
    model = DeepFM(linear_feature_columns, dnn_feature_columns,
                   task='binary', device=device)
elif model_name == "din":
    model = DIN(linear_feature_columns, dnn_feature_columns,
                task='binary', device=device)
elif model_name == "wdl":
    model = WDL(linear_feature_columns, dnn_feature_columns,
                task='binary', device=device)
elif model_name == "dcn":
    model = DCN(linear_feature_columns, dnn_feature_columns,
                task='binary', device=device)
elif model_name == "nfm":
    model = NFM(linear_feature_columns, dnn_feature_columns,
                task='binary', device=device)

import datetime
xmh_model_dir = "xmh_logs/" + args.dataset + "-" + model_name + "-" + optimizer_dense +str(optimizer_dense_lr) + str(optimizer_sparse) + str(optimizer_sparse_lr)

model.compile(optimizer=optimizer_dense, loss="binary_crossentropy",
              metrics=['binary_crossentropy', 'acc', 'AUC'],
              optimizer_sparse=optimizer_sparse,
              optimizer_dense_lr=optimizer_dense_lr,
              optimizer_sparse_lr=optimizer_sparse_lr, )
if args.debug:
    verbose_steps = 10
else:
    verbose_steps = 20000
history = model.fit(train_model_input, train[target].values,
                    batch_size=256, epochs=1, verbose=1, validation_split=0.1, model_name=model_name,
                   verbose_steps= verbose_steps, xmh_model_dir = xmh_model_dir)
pred_ans = model.predict(test_model_input, batch_size=256)
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test Accuracy", round(accuracy_score(
    test[target].values, pred_ans > 0.5), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))


# In[ ]:





# In[21]:


# In[23]:



# In[ ]:





