import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _EmbeddingColumn

from .utils import LINEAR_SCOPE_NAME, variable_scope, get_collection, get_GraphKeys, input_layer, get_losses
from .. import KUIBA

if KUIBA:
    from kraken import (KrakenHook, get_kraken_trainable_embeddings,
                       register_variable, shared_embedding_columns, KrakenDataset,
                       kraken_init, generate_ps_config, update_all_variables, kraken_push_auc, get_kraken_embeddings)

def linear_model(features, linear_feature_columns):
    if tf.__version__ >= '2.0.0':
        linear_logits = tf.compat.v1.feature_column.linear_model(features, linear_feature_columns)
    elif tf.__version__ <= '1.13.0':
        if KUIBA:
            from kraken import LinearModel
            linear_model_t = LinearModel(linear_feature_columns)
        else:
            from tensorflow.python.feature_column import feature_column_v2 as fc
            linear_model_t = fc.LinearModel(linear_feature_columns)
        linear_logits = linear_model_t(features)
    else:
        linear_logits = tf.feature_column.linear_model(features, linear_feature_columns)
    return linear_logits


def get_linear_logit(features, linear_feature_columns, l2_reg_linear=0):
    with variable_scope(LINEAR_SCOPE_NAME):
        if not linear_feature_columns:
            linear_logits = tf.Variable([[0.0]], name='bias_weights')
        else:

            linear_logits = linear_model(features, linear_feature_columns)

            if l2_reg_linear > 0:
                for var in get_collection(get_GraphKeys().TRAINABLE_VARIABLES, LINEAR_SCOPE_NAME)[:-1]:
                    get_losses().add_loss(l2_reg_linear * tf.nn.l2_loss(var, name=var.name.split(":")[0] + "_l2loss"),
                                          get_GraphKeys().REGULARIZATION_LOSSES)
    return linear_logits


def input_from_feature_columns(features, feature_columns, l2_reg_embedding=0.0):
    dense_value_list = []
    sparse_emb_list = []
    for feat in feature_columns:
        print(feat)
        if is_embedding(feat):
            sparse_emb = tf.expand_dims(input_layer(features, [feat]), axis=1)
            sparse_emb_list.append(sparse_emb)
            if l2_reg_embedding > 0:
                get_losses().add_loss(l2_reg_embedding * tf.nn.l2_loss(sparse_emb, name=feat.name + "_l2loss"),
                                      get_GraphKeys().REGULARIZATION_LOSSES)

        else:
            dense_value_list.append(input_layer(features, [feat]))

    return sparse_emb_list, dense_value_list


def is_embedding(feature_column):
    temp = None
    if KUIBA:
        import kraken
        temp = kraken.feature_column.KrakenSharedEmbeddingColumn
    try:
        from tensorflow.python.feature_column.feature_column_v2 import EmbeddingColumn
    except:
        EmbeddingColumn = _EmbeddingColumn
    if KUIBA:
        return isinstance(feature_column, (_EmbeddingColumn, EmbeddingColumn, temp))
    else:
        return isinstance(feature_column, (_EmbeddingColumn, EmbeddingColumn))
