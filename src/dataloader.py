
import pickle
import queue
import numpy as np
#import progressbar

import src.config as config

def get_train_data(root_data, start):
    minibatch_x_root = np.zeros(shape=[config.batch_size, config.in_seq_length, config.road_num])
    minibatch_y_root = np.zeros(shape=[config.batch_size, config.out_seq_length, config.road_num])

    for i in range(config.batch_size):
        minibatch_x_root[i] = root_data[start+i : start+i+config.in_seq_length]
        minibatch_y_root[i] = root_data[start+i+config.in_seq_length+config.pred_time-1 : start+i+config.in_seq_length+config.pred_time-1+config.out_seq_length]

    minibatch_decode_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))
    minibatch_target_seq = np.zeros((minibatch_y_root.shape[0], minibatch_y_root.shape[1] + 1, minibatch_y_root.shape[2]))

    minibatch_decode_seq[:, 1:, :] = minibatch_y_root
    minibatch_target_seq[:, :-1, :] = minibatch_y_root

    minibatch_decode_seq[:, 0, :] = config.start_id
    minibatch_target_seq[:, -1, :] = config.end_id

    return minibatch_x_root, minibatch_decode_seq, minibatch_target_seq

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


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



if __name__ == "__main__":
    # find_neighbours(5, 5)
    # r, n, p = load_data(5, 5)
    # get_minibatch(r, n, order=[0,1], num_seq=r.shape[1] - (config.in_seq_length + config.out_seq_length) + 1)
    '''
    r = load_data_all()
    import time
    st = time.time()
    get_minibatch_all(r, order=list(range(config.batch_size)), num_seq=r.shape[1] - (config.in_seq_length + config.out_seq_length) + 1)
    print(time.time() - st)
    '''
    '''
    fi, ft, fp = load_features(["1525826704", "1561981475"])
    print(fi)
    print(ft)
    print(fp)
    '''
    # load_features()
    '''
    e = load_event_data()
    for node in e.keys():
        road = node
        break
    print(e[road])
    ef = get_event_filter(e[road])
    print(ef)
    print(len(ef))
    exit()
    x, d, t, eidx, end = get_minibatch_4_test_event(r[:, -config.valid_length:,:], ef, 0, 0)
    print(x.shape)
    print(d.shape)
    print(t.shape)
    print(eidx)
    print(end)
    '''
    '''
    e = load_event_data()
    p = get_pathlist()
    eap = get_event_filter_allpath(e, p)

    each_num_seq = config.valid_length - (config.in_seq_length + config.out_seq_length) + 1
    total_batch_size = 15073 * each_num_seq

    eorder = get_event_orders(eap, list(range(total_batch_size)), each_num_seq)
    print(len(eorder))

    for i in range(10):
        pathid = eorder[i] // each_num_seq
        pathlod = eorder[i] % each_num_seq
        print(pathid)
        print(pathlod)

        print(eap[pathid][pathlod + config.in_seq_length: pathlod + config.in_seq_length + config.out_seq_length])
    '''
    get_query_data()



    pass

