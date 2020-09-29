import sys  # nopep8
sys.path.append("../")  # nopep8


import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.estimator import DeepFMEstimator
from deepctr.estimator.inputs import input_fn_pandas

import random
import numpy as np

if __name__ == "__main__":
    tf.reset_default_graph()
    tf.set_random_seed(0)
    random.seed(0)
    np.random.seed(0)

    tf.logging.set_verbosity(tf.logging.INFO)
    data = pd.read_csv(
        '/data/project/deep-ctr-torch/avazu-ctr-prediction/add_ts_mini.csv')

    sparse_features = 'C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21'.split(
        ',')
    dense_features = []

    data[sparse_features] = data[sparse_features].fillna('-1', )
    # data[dense_features] = data[dense_features].fillna(0, )
    target = 'click'

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # mms = MinMaxScaler(feature_range=(0, 1))
    # data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    dnn_feature_columns = []
    linear_feature_columns = []

    for i, feat in enumerate(sparse_features):
        dnn_feature_columns.append(tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(feat, data[feat].nunique()), 4))
        linear_feature_columns.append(
            tf.feature_column.categorical_column_with_identity(feat, data[feat].nunique()))
    for feat in dense_features:
        dnn_feature_columns.append(tf.feature_column.numeric_column(feat))
        linear_feature_columns.append(tf.feature_column.numeric_column(feat))

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.1, shuffle=False)

    # Not setting default value for continuous feature. filled with mean.

    train_model_input = input_fn_pandas(
        train, sparse_features + dense_features, target, batch_size=128)
    test_model_input = input_fn_pandas(
        test, sparse_features + dense_features, None, batch_size=128)

    # 4.Define Model,train,predict and evaluate

    adam = tf.train.AdamOptimizer()

    # mirrored_strategy = tf.distribute.MirroredStrategy()

    model = DeepFMEstimator(linear_feature_columns,
                            dnn_feature_columns, dnn_hidden_units=(256, 128), task='binary',
                            dnn_optimizer=adam, linear_optimizer=adam, model_dir="tmp")

    model.train(train_model_input)
    pred_ans_iter = model.predict(test_model_input)
    pred_ans = list(map(lambda x: x['pred'], pred_ans_iter))
    # print(pred_ans)
    #
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
