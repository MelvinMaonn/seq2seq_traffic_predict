
batch_size      = 128
#in_seq_length   = 4 * 24
in_seq_length   = 12
#out_seq_length  = 4 * 2
out_seq_length  = 1
num_neighbour   = 10

# short pred(n<12) is the 11+n, long pred is always 11+1
pred_time = 11+1

dim_hidden = 2048
query_dim_hidden = 128
# dim_hidden      = 64
# query_dim_hidden = 32

dim_features_info = 131
dim_features_time = 6
dim_features    = dim_features_info + dim_features_time

road_num = 829

full_length     = 61 * 24 * 4
valid_length    = 2900

start_id        = 100
pad_id          = 0
end_id          = 101

epoch           = 200
save_p_epoch    = 5
test_p_epoch    = 5

# TRAIN_DATA_PATH = '../data/800r_train_2.txt'
# VAL_DATA_PATH = '../data/800r_test.txt'
# TRAIN_DATA_PATH = '/mnt/data1/zll/mm/data/seq2seq_data/seq2seq/train_7_9_weekday_1_12.npz'
TRAIN_SHUFFLE_DATA_PATH = '/mnt/data1/zll/mm/data/seq2seq_data/seq2seq/train_shuffle_7_9_weekday_1_12.npz'
# VAL_DATA_PATH = '/mnt/data1/zll/mm/data/seq2seq_data/seq2seq/val_7_9_weekday_1_12.npz'
# TRAIN_DATA_PATH = '/mnt/data1/zll/mm/data/seq2seq_data/train_short.csv'
# VAL_DATA_PATH = '/mnt/data1/zll/mm/data/seq2seq_data/test_short.csv'
# TRAIN_DATA_PATH = '/mnt/data1/zll/mm/data/seq2seq_data/train_20min.csv'
# VAL_DATA_PATH = '/mnt/data1/zll/mm/data/seq2seq_data/test_20min.csv'
# TRAIN_DATA_PATH = '/mnt/data1/zll/mm/data/seq2seq_data/train_30min.csv'
# VAL_DATA_PATH = '/mnt/data1/zll/mm/data/seq2seq_data/test_30min.csv'
# TRAIN_DATA_PATH = '/mnt/data1/zll/mm/data/seq2seq_data/train_40min.csv'
# VAL_DATA_PATH = '/mnt/data1/zll/mm/data/seq2seq_data/test_40min.csv'
TRAIN_DATA_PATH = '/mnt/data1/zll/mm/data/seq2seq_data/train_60min.csv'
VAL_DATA_PATH = '/mnt/data1/zll/mm/data/seq2seq_data/test_60min.csv'

data_path       = "../../data/"
result_path     = "../results/"

model_path      = "../models/"
logs_path       = "../logs/"
figs_path       = "../figs/"

import src.utils as utils
global_start_time = utils.now2string()

import numpy as np
np.set_printoptions(
    linewidth=150,
    formatter={'float_kind': lambda x: "%.4f" % x}
)

impact_k        = 150
# 150 epoch 20 for query_comb
# 300 epoch 30 for query_comb

# all_model_stage_epoch = [100 + 1, 150 + 1]
all_model_stage_epoch = [100 + 1, 130 + 1]
# all_model_stage_epoch = [30 + 1, 40 + 1]

gpu_memory      = 1

