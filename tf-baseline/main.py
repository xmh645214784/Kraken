import sys  # nopep8
sys.path.append("./DeepCTR/")  # nopep8
import os
import getpass
import argparse
import tensorflow as tf
import time
from deepctr.estimator import DeepFMEstimator, WDLEstimator, DCNEstimator
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import feature_column as feature_column


MAX_FEATURE_NUM = 30*1e6
WORKER_NUM = 1
#

DATA_PREFIX = "/data/project/deep-ctr-torch/"


def load_criteo(params):
    # Data processing code adapted from https://github.com/facebookresearch/dlrm
    # Follow steps in https://github.com/ylongqi/dlrm/blob/master/data_utils.py to generate kaggle_processed.npz
    # Or using `./download_dataset.sh criteo` command to download the processed data.
    print("------------------------------------")
    import os
    dataset_folder = DATA_PREFIX + "criteo/"
    if params.debug:
        datapath = dataset_folder + 'kaggle_processed_tiny.npz'
    else:
        datapath = dataset_folder + 'kaggle_processed.npz'
    import numpy as np

    with np.load(datapath) as data:
        X_int = data["X_int"]
        X_cat = data["X_cat"]
        y = data["y"]
        counts = data["counts"]

    indices = np.arange(len(y))
    indices = np.array_split(indices, 7)
    for i in range(len(indices)):
        indices[i] = np.random.permutation(indices[i])

    train_indices = np.concatenate(indices[:-1])
    test_indices = indices[-1]
    val_indices, test_indices = np.array_split(test_indices, 2)
    train_indices = np.random.permutation(train_indices)

    raw_data = dict()

    raw_data['counts'] = counts
    raw_data['X_cat_train'] = X_cat[train_indices].astype(np.int32)
    raw_data['X_int_train'] = np.log(X_int[train_indices]+1).astype(np.float32)
    raw_data['y_train'] = y[train_indices].astype(np.float32)

    raw_data['X_cat_val'] = X_cat[val_indices].astype(np.int32)
    raw_data['X_int_val'] = np.log(X_int[val_indices]+1).astype(np.float32)
    raw_data['y_val'] = y[val_indices].astype(np.float32)

    raw_data['X_cat_test'] = X_cat[test_indices].astype(np.int32)
    raw_data['X_int_test'] = np.log(X_int[test_indices]+1).astype(np.float32)
    raw_data['y_test'] = y[test_indices].astype(np.float32)

    return raw_data


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        if self.iterator_initializer_func:
            self.iterator_initializer_func(session)


def get_feature_columns(param):
    def find_bucket_size(ln_embedding, max_feature_num=MAX_FEATURE_NUM):
        left = 0
        right = max(ln_embedding)
        while left < right:
            mid = (left + right) // 2
            x = sum([
                each if each < mid else mid for each in ln_embedding])
            if x < max_feature_num:
                left = mid + 1
            elif x == max_feature_num:
                break
            else:
                right = mid - 1
        return mid

    def criteo_get_fc():
        ln_embedding = [1461,      584, 10131227,  2202608,      306,       24,
                        12518,      634,        4,    93146,     5684,  8351593,
                        3195,       28,    14993,  5461306,       11,     5653,
                        2173,        4,  7046547,       18,       16,   286181,
                        105,   142572]

        param.max_bucket = find_bucket_size(ln_embedding)
        print("TensorFlow, max_bucket_size = ", param.max_bucket)
        if param.max_bucket is not None:
            ln_embedding = [
                each if each < param.max_bucket else param.max_bucket for each in ln_embedding]
        dense = []
        for i in range(13):
            dense.append(fc.numeric_column("I{}".format(i),
                                           dtype=tf.int64, default_value=0))

        dnn_feature_columns = []
        linear_feature_columns = []

        dnn_feature_columns += dense
        linear_feature_columns  += dense

        sparse_emb = []
        for i in range(26):
            linear_feature_columns.append(fc.categorical_column_with_hash_bucket(
                "C{}".format(i), ln_embedding[i], dtype=tf.int64))
            ids = fc.categorical_column_with_hash_bucket(
                "C{}".format(i), ln_embedding[i], dtype=tf.int64)
            sparse_emb += [fc.embedding_column(ids, param.embedding_size)]
        dnn_feature_columns += sparse_emb
        return dnn_feature_columns, linear_feature_columns

    def avazu_get_fc():
        # nr samples 32747463
        ln_embedding = [40428967, 7, 7, 4737, 7745, 26, 8552, 559, 36,
                        2686408, 6729486, 8251, 5, 4, 2626, 8, 9, 435, 4, 68, 172, 60]
        param.max_bucket = find_bucket_size(ln_embedding)
        print("TensorFlow, max_bucket_size = ", param.max_bucket)
        if param.max_bucket is not None:
            ln_embedding = [
                each if each < param.max_bucket else param.max_bucket for each in ln_embedding]

        dnn_feature_columns = []
        linear_feature_columns = []

        sparse_features = ['id', 'C1', 'banner_pos', 'site_id', 'site_domain',
                           'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
                           'device_ip', 'device_model', 'device_type', 'device_conn_type', ] \
            + ['C' + str(i) for i in range(14, 22)]

        sparse_emb = []
        for i in range(len(sparse_features)):
            linear_feature_columns.append(fc.categorical_column_with_hash_bucket(
              sparse_features[i], ln_embedding[i], dtype=tf.string))
            ids = fc.categorical_column_with_hash_bucket(
              sparse_features[i], ln_embedding[i], dtype=tf.string)
            sparse_emb += [fc.embedding_column(ids, param.embedding_size)]

        dnn_feature_columns += sparse_emb

        return dnn_feature_columns, linear_feature_columns

    def movielen_get_fc():
        sparse_features = ["movieId", "userId", 'imdbId', 'tmdbId', 'genres']
        # movieId 59047
        # userId 162541
        # imdbId 59047
        # tmdbId 61342
        # genres 21
        ln_embedding = [59047, 162541, 59047, 61342, 21]

        param.max_bucket = find_bucket_size(ln_embedding)
        print("TensorFlow, max_bucket_size = ", param.max_bucket)
        if param.max_bucket is not None:
            ln_embedding = [
                each if each < param.max_bucket else param.max_bucket for each in ln_embedding]

        dnn_feature_columns = []
        linear_feature_columns = []

        for i, feat in enumerate(sparse_features):
            dnn_feature_columns.append(fc.embedding_column(
                fc.categorical_column_with_hash_bucket(feat, ln_embedding[i], dtype=tf.string), 4))
            linear_feature_columns.append(
                fc.categorical_column_with_hash_bucket(feat, ln_embedding[i], dtype=tf.string))

        return dnn_feature_columns, linear_feature_columns

    if param.dataset == "criteo":
        return criteo_get_fc()
    elif param.dataset == "avazu":
        return avazu_get_fc()
    elif param.dataset == "movielen":
        return movielen_get_fc()
    else:
        raise Exception("invalid dataset")


def input_fn(data_set, mode, params, worker_id, strategy=None):
    # total 39291960 samples
    iterator_initializer_hook = IteratorInitializerHook()

    def criteo_train_inputs():
        with tf.name_scope('Training_data'):
            feed_slice_dict = dict()
            for i in range(13):
                if mode == 'train':
                    feed_slice_dict['I{}'.format(
                        i)] = data_set['X_int_train'][:, i]
                elif mode == 'eval':
                    feed_slice_dict['I{}'.format(
                        i)] = data_set['X_int_val'][:, i]

            for i in range(26):
                if mode == 'train':
                    feed_slice_dict['C{}'.format(
                        i)] = data_set['X_cat_train'][:, i]
                elif mode == 'eval':
                    feed_slice_dict['C{}'.format(
                        i)] = data_set['X_cat_val'][:, i]

            if mode == 'train':
                feed_slice_dict['label'] = data_set['y_train']
                dataset_len = len(data_set['y_train'])
            elif mode == 'eval':
                feed_slice_dict['label'] = data_set['y_val']
                dataset_len = len(data_set['y_val'])

            slice_dict = dict([(k, tf.placeholder(v.dtype, v.shape, name=k))
                               for k, v in feed_slice_dict.items()])
            feed_slice_dict = dict([('Training_data/'+k+':0', v)
                                    for k, v in feed_slice_dict.items()])
            dataset = tf.data.Dataset.from_tensor_slices(slice_dict)
            dataset = take_dataset(dataset, dataset_len)
            iterator = dataset.make_initializable_iterator()
            next_example = iterator.get_next()
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict=feed_slice_dict)
            label = next_example['label']
            next_example.pop("label")
            return next_example, label

    def take_dataset(dataset, dataset_len):
        import os
        import time
        global_num = 1 * WORKER_NUM
        global_id = 0 * WORKER_NUM + worker_id

        if mode == tf.estimator.ModeKeys.EVAL:
            pass
        else:
            # [dataset_len//global_num*global_id, dataset_len//global_num*(id+1)]
            take_len = (dataset_len+global_num-1)//global_num
            take_start = take_len * global_id
            dataset = dataset.skip(take_start)
            dataset = dataset.take(take_len)
        dataset = dataset.batch(params.batch_size)
        dataset = dataset.prefetch(10000)
        return dataset

    def avazu_train_inputs():
        # Extract lines from input files using the Dataset API.
        # can pass one filename or filename list
        def avazu_parse_csv(line):
            # There are 13 integer features and 26 categorical features
            LABEL_COLUMN = "click"
            CSV_COLUMNS = ['id', 'click', 'C1', 'banner_pos', 'site_id', 'site_domain',
                           'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
                           'device_ip', 'device_model', 'device_type', 'device_conn_type', ] \
                + ['C' + str(i) for i in range(14, 22)] + ["ts"]
            # Columns Defaults
            CSV_COLUMN_DEFAULTS = [tf.constant(
                [], dtype=tf.string) for each in range(len(CSV_COLUMNS))]
            CSV_COLUMN_DEFAULTS[1] = tf.constant([], dtype=tf.int64)
            CSV_COLUMN_DEFAULTS[-1] = tf.constant([], dtype=tf.int64)

            columns = tf.decode_csv(line, record_defaults=CSV_COLUMN_DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            labels = features.pop(LABEL_COLUMN)
            return features, labels
        import os
        p = ""
        if params.debug:
            if mode == tf.estimator.ModeKeys.TRAIN:
                p = DATA_PREFIX + "avazu-ctr-prediction/add_ts_mini_train.csv"
            else:
                p = DATA_PREFIX + "avazu-ctr-prediction/add_ts_mini_eval.csv"
        else:
            if mode == tf.estimator.ModeKeys.TRAIN:
                p = DATA_PREFIX + "avazu-ctr-prediction/add_ts_train.csv"
            else:
                p = DATA_PREFIX + "avazu-ctr-prediction/add_ts_eval.csv"
        dataset_len = int(
            os.popen(f"wc -l {p} |awk '{{print $1}}'").readlines()[0]) - 1
        dataset = tf.data.TextLineDataset(p).skip(1)
        dataset = dataset.map(
            avazu_parse_csv, num_parallel_calls=20)
        dataset = take_dataset(dataset, dataset_len)

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    def movie_len_train_inputs():
        def process_list_column(list_column):
            sparse_strings = tf.string_split(list_column, delimiter="|")
            return sparse_strings

        def parse_csv(value):
            CSV_COLUMNS = ['nouse', 'userId', 'movieId', 'rating',
                           'timestamp', 'title', 'genres', 'imdbId', 'tmdbId']
            CSV_COLUMN_DEFAULTS = [[0], [''], [''],
                                   [0], [0], [''], [''], [''], ['']]
            columns = tf.decode_csv(value, record_defaults=CSV_COLUMN_DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            features['genres'] = process_list_column([features['genres']])
            label = features.pop("rating")
            return features, label

        if params.debug:
            if mode == tf.estimator.ModeKeys.TRAIN:
                p = DATA_PREFIX + "ml_25m/ml_25m_test-mini_train.csv"
            else:
                p = DATA_PREFIX + "ml_25m/ml_25m_test-mini_eval.csv"
        else:
            if mode == tf.estimator.ModeKeys.TRAIN:
                p = DATA_PREFIX + "ml_25m/ml_25m_test_train.csv"
            else:
                p = DATA_PREFIX + "ml_25m/ml_25m_test_eval.csv"
        dataset = tf.data.TextLineDataset(p).skip(1)
        dataset = dataset.map(parse_csv, num_parallel_calls=20)
        dataset_len = int(
            os.popen(f"wc -l {p} |awk '{{print $1}}'").readlines()[0]) - 1
        dataset = take_dataset(dataset, dataset_len)
        iterator = dataset.make_one_shot_iterator()
        features, label = iterator.get_next()
        return features, label

    print(params.dataset)
    if params.dataset == "criteo":
        return criteo_train_inputs, iterator_initializer_hook
    elif params.dataset == "avazu":
        return avazu_train_inputs, iterator_initializer_hook
    elif params.dataset == "movielen":
        return movie_len_train_inputs, iterator_initializer_hook
    else:
        assert False


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--model-dir", default="/dev/shm/checkpoint")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--embedding-size", default=4, type=int)
    parser.add_argument('--optimizer', type=str, choices=["adam", "sgd", "adagrad"])
    parser.add_argument("--hidden-units", default=[256, 128])
    parser.add_argument("--model", type=str, default="dnn",
                        choices=["dcn", "wdl", "dnn", "deepfm", "lr"])
    parser.add_argument("--dataset", type=str,
                        choices=["criteo", "avazu", "movielen"])
    parser.add_argument("--tb_size", type=float, default="0.6")
    parser.add_argument("--learning-rate", type=float, default=None)

    params = parser.parse_args()
    if params.dataset == "criteo":
        data = load_criteo(params)
    elif params.dataset == "avazu":
        data = None
    elif params.dataset == "amazon":
        data = None
    elif params.dataset == "movielen":
        data = None
    else:
        raise Exception("invalid dataset")

    if params.debug:
        params.log_step_count_steps = 5
        params.save_checkpoints_steps = 5
        params.save_summary_steps = 5
    else:
        params.log_step_count_steps = 1000
        params.save_checkpoints_steps = 50000
        params.save_summary_steps = 1000
    params.save_checkpoints_secs = None
    params.keep_checkpoint_max = 3

    params.max_bucket = None
    params.flag_file = '../config/server_static.flags'

    if params.optimizer == "adam":
        if params.learning_rate == None:
            params.learning_rate = 0.001
        linear_opt = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        dnn_opt = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    elif params.optimizer == "sgd":
        if params.learning_rate == None:
            params.learning_rate = 0.01
        linear_opt = tf.train.GradientDescentOptimizer(params.learning_rate)
        dnn_opt = tf.train.GradientDescentOptimizer(params.learning_rate)
    elif params.optimizer == "adagrad":
        if params.learning_rate == None:
            params.learning_rate = 0.001
        linear_opt = tf.train.AdagradOptimizer(params.learning_rate)
        dnn_opt = tf.train.AdagradOptimizer(params.learning_rate)


    dnn_feature_columns, linear_feature_columns = get_feature_columns(params)
    if params.model == "dcn":
        T = DCNEstimator
    elif params.model == "wdl":
        T = WDLEstimator
    elif params.model == "dnn":
        T = WDLEstimator
        linear_feature_columns = []
    elif params.model == "deepfm":
        T = DeepFMEstimator

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        session_config=session_config,
        log_step_count_steps=params.log_step_count_steps,
        save_checkpoints_steps=params.save_checkpoints_steps,
        save_checkpoints_secs=params.save_checkpoints_secs,
        keep_checkpoint_max=params.keep_checkpoint_max,
        save_summary_steps=params.save_summary_steps)

    model = T(linear_feature_columns, dnn_feature_columns,
              model_dir=params.model_dir,
              task='binary',
              dnn_hidden_units=[256, 128], config=run_config,
              dnn_optimizer=dnn_opt, linear_optimizer=linear_opt)

    import os
    try:
        import shutil
        shutil.rmtree(params.model_dir)
    except:
        pass

    train_input_fn, train_input_hook = input_fn(
        data, mode='train', params=params, worker_id=0, )
    eval_input_fn, eval_input_hook = input_fn(
        data, mode='eval', params=params, worker_id=0)

    model.train(train_input_fn, hooks=[train_input_hook])
    print("train finished")
    model.evaluate(eval_input_fn, hooks=[eval_input_hook])
    print("evalutor finished")
