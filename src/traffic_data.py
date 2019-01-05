import numpy as np
from numpy.random import shuffle
from sklearn.utils import shuffle

from src import config


def get_all_traffic_data():
    data_path = '/mnt/data1/zll/mm/data/seq2seq_data/beijing_829_60_71'

    train_data = np.load(data_path + '/train.npz')
    train_data_x = train_data['x'][..., 0]  # shape:(1725, 12, 829)
    train_data_y = train_data['y'][..., 0]  # shape:(1725, 12, 829)

    train_data_x, train_data_y = shuffle(train_data_x, train_data_y)

    # val_data = np.load(data_path + '/val.npz')
    # val_data_x = val_data['x'][..., 0]  # shape:(215, 12, 829)
    # val_data_y = val_data['y'][..., 0]  # shape:(215, 12, 829)

    # test_data = np.load(data_path + '/test.npz')
    # test_data_x = test_data['x'][..., 0]  # shape:(196, 12, 829)
    # test_data_y = test_data['y'][..., 0]  # shape:(196, 12, 829)
    # print(train_data_x.shape, train_data_y.shape,
    #       val_data_x.shape, val_data_y.shape,
    #       test_data_x.shape, test_data_y.shape)

    # data_x = np.append(train_data_x, val_data_x, axis=0)  # shape:(1940, 12, 829)
    # data_y = np.append(train_data_y, val_data_y, axis=0)  # shape:(1940, 12, 829)
    # print(data_x.shape, data_y.shape)

    # data_x = np.append(data_x, test_data_x, axis=0)  # shape:(2136, 12, 829)
    # data_y = np.append(data_y, test_data_y, axis=0)  # shape:(2136, 12, 829)
    # print(data_x.shape, data_y.shape)

    np.savez_compressed('/mnt/data1/zll/mm/data/seq2seq_data/seq2seq' + '/train_shuffle_7_9_weekday_60_71.npz', x=train_data_x, y=train_data_y)
    # np.savez_compressed('seq2seq' + '/val_7_9_weekday_60_71.npz', x=val_data_x, y=val_data_y)
    # np.savez_compressed('seq2seq' + '/test_7_9_weekday_60_71.npz', x=test_data_x, y=test_data_y)


def read_data_sample():
    data_path = '/mnt/data1/zll/mm/data/seq2seq_data/seq2seq/val_7_9_weekday_1_12.npz'
    data = np.load(data_path)
    data_x_segments = data['x']  # shape:(2136, 12, 829)
    data_y_segments = data['y']  # shape:(2136, 12, 829)

    num_samples, num_time, num_nodes = data_x_segments.shape
    print(num_samples, num_time, num_nodes)
    for i in range(num_nodes):
        data_x = data_x[..., i]  # shape:(2136, 12)
        data_y = data_y[..., i]  # shape:(2136, 12)
        print(data_x.shape, data_y.shape)
        for j in range(num_samples):
            print(data_x[j, :], data_y[j, :])  # 1-12min
            print(data_x[j, :], data_y[j, 1])  # 2min
            print(data_x[j, :], data_y[j, 3])  # 4min
            print(data_x[j, :], data_y[j, 5])  # 6min
            print(data_x[j, :], data_y[j, 7])  # 8min
            print(data_x[j, :], data_y[j, 11])  # 12min

    '''
    for i in range(2):  # 只打印两个路段
        data_x = data_x_segments[..., i]  # shape:(2136, 12)
        data_y = data_y_segments[..., i]  # shape:(2136, 12)
        print(data_x.shape, data_y.shape)
        for j in range(10):  # 只打印10个
            print(data_x[j, :], data_y[j, :])  # 1-12min
            print(data_x[j, :], data_y[j, 1])  # 2min
            print(data_x[j, :], data_y[j, 3])  # 4min
            print(data_x[j, :], data_y[j, 5])  # 6min
            print(data_x[j, :], data_y[j, 7])  # 8min
            print(data_x[j, :], data_y[j, 11])  # 12min
    '''

def get_data(root_data, start):
    minibatch_x_root = root_data['x'][start:start+config.batch_size]
    minibatch_y_root = root_data['y'][start:start+config.batch_size, config.pred_time-1:config.pred_time, :]

    minibatch_decode_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_target_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))

    minibatch_decode_seq[:, 1:, :] = minibatch_y_root
    minibatch_target_seq[:, :-1, :] = minibatch_y_root

    minibatch_decode_seq[:, 0, :] = config.start_id
    minibatch_target_seq[:, -1, :] = config.end_id

    return minibatch_x_root, minibatch_decode_seq, minibatch_target_seq


if __name__ == "__main__":
    # get_all_traffic_data()
    read_data_sample()

    # data_path = '/mnt/data1/zll/mm/data/seq2seq_data/seq2seq/train_7_9_weekday_1_12.npz'
    # data = np.load(data_path)
    #
    # get_data(data, 0)
