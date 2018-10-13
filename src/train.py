
import os
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
# import seaborn as sns
import time
import numpy as np
import sklearn
import random
import sklearn.utils
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import threading_data
from datetime import datetime, timedelta

import src.log as log
import src.model as model
import src.utils as utils
import src.config as config
import src.dataloader as dataloader

os.environ['CUDA_VISIBLE_DEVICES'] = '4'


class Controller():

    def __init__(self, model, base_epoch):
        self.model = model
        self.saver = tf.train.Saver(max_to_keep=50)
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.per_process_gpu_memory_fraction = config.gpu_memory
        self.sess = tf.Session(config=gpu_config)
        tl.layers.initialize_global_variables(self.sess)

        self.base_epoch = base_epoch
        self.__init_path__()
        self.__init_mkdir__()

    def save_model(self, path, global_step=None):
        save_path = self.saver.save(self.sess, path, global_step=global_step)
        print("[S] Model saved in ckpt %s" % save_path)
        return save_path

    def restore_model(self, path, global_step=None):
        model_path = "%s-%s" % (path, global_step)
        self.saver.restore(self.sess, model_path)
        print("[R] Model restored from ckpt %s" % model_path)
        return True

    def __init_path__(self):
        self.model_save_dir = config.model_path + self.model.model_name + "/"
        self.log_save_dir = config.logs_path + self.model.model_name + "/"
        self.figs_save_dir = config.figs_path + self.model.model_name + "/"

    def __init_mkdir__(self):
        dirlist = [
            self.model_save_dir,
            self.log_save_dir,
            self.figs_save_dir
        ]
        utils.make_dirlist(dirlist)

    def __train__(self, epoch, root_data, neighbour_data, logger):

        each_num_seq = root_data.shape[1] - (config.in_seq_length + config.out_seq_length) + 1
        total_batch_size = root_data.shape[0] * each_num_seq
        train_order = list(range(total_batch_size))
        random.shuffle(train_order)
        train_order = train_order[:config.batch_size * 1000]

        all_loss = np.zeros(7)

        start_time = time.time()
        step_time = time.time()
        train_steps = len(train_order) // config.batch_size

        for cstep in range(train_steps):

            x_root, x_neight, decode_seq, target_seq = dataloader.get_minibatch(
                root_data,
                neighbour_data,
                order=train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size],
                num_seq=each_num_seq
            )

            global_step = cstep + epoch * train_steps

            results = self.sess.run([
                self.model.mae_copy,
                self.model.train_loss,
                self.model.nmse_train_loss,
                self.model.nmse_train_noend,
                self.model.rmse_train_noend,
                self.model.mae_train_noend,
                self.model.mape_train_noend,
                self.model.learning_rate,
                self.model.optim],
                feed_dict={
                    self.model.x_root: x_root,
                    self.model.x_neighbour: x_neight,
                    self.model.decode_seqs: decode_seq,
                    self.model.target_seqs: target_seq,
                    self.model.global_step: global_step,
                })

            all_loss += np.array(results[:-2])

            if cstep % 100 == 0 and cstep > 0:
                print(
                    "[Train] Epoch: [%3d][%4d/%4d] time: %.4f, lr: %.8f, loss: %s" %
                    (epoch, cstep, train_steps, time.time() - step_time, results[-2], all_loss / (cstep + 1))
                )
                step_time = time.time()
                logger.add_log(global_step, all_loss / (cstep + 1))

        print(
            "[Train Sum] Epoch: [%3d] time: %.4f, lr: %.8f, loss: %s" %
            (epoch, time.time() - start_time, results[-2], all_loss / train_steps)
        )
        logger.add_log(global_step, all_loss / train_steps)

        return all_loss

    def __valid__(self, epoch, root_data, neighbour_data, logger):

        each_num_seq = root_data.shape[1] - (config.in_seq_length + config.out_seq_length) + 1
        total_batch_size = root_data.shape[0] * each_num_seq
        train_order = list(range(total_batch_size))

        all_loss = np.zeros(7)

        start_time = time.time()
        step_time = time.time()
        valid_steps = total_batch_size // config.batch_size

        for cstep in range(valid_steps):

            x_root, x_neight, decode_seq, target_seq = dataloader.get_minibatch(
                root_data,
                neighbour_data,
                order=train_order[cstep * config.batch_size : (cstep + 1) * config.batch_size],
                num_seq=each_num_seq
            )

            results = self.sess.run([
                self.model.mae_copy,
                self.model.test_loss,
                self.model.nmse_test_loss,
                self.model.nmse_test_noend,
                self.model.rmse_test_noend,
                self.model.mae_test_noend,
                self.model.mape_test_noend],
                feed_dict={
                    self.model.x_root: x_root,
                    self.model.x_neighbour: x_neight,
                    self.model.decode_seqs: decode_seq,
                    self.model.target_seqs: target_seq,
                })

            all_loss += np.array(results[:])

            if cstep % 100 == 0 and cstep > 0:
                print(
                    "[Valid] Epoch: [%3d][%4d/%4d] time: %.4f, loss: %s" %
                    (epoch, cstep, valid_steps, time.time() - step_time, all_loss / (cstep + 1))
                )
                step_time = time.time()

        print(
            "[Valid Sum] Epoch: [%3d] time: %.4f, loss: %s" %
            (epoch, time.time() - start_time, all_loss / valid_steps)
        )
        logger.add_log(epoch, all_loss / valid_steps)

        return all_loss

    def __test__(self, epoch, root_data, neighbour_data, logger, pathlist, test_interval=10):

        each_num_seq = root_data.shape[1] - (config.in_seq_length + config.out_seq_length) + 1
        total_batch_size = root_data.shape[0] * each_num_seq
        train_order = list(range(total_batch_size))

        all_loss = np.zeros(1)
        time_loss = np.zeros(config.out_seq_length)

        start_time = time.time()
        step_time = time.time()

        round = 0
        pathpred = list()
        for path in range(0, root_data.shape[0], test_interval):
            predlist = list()
            step_time = time.time()
            for cstep in range(each_num_seq // config.batch_size):
                round += 1

                x_root, x_neight, decode_seq, target_seq = dataloader.get_minibatch_4_test_neighbour(
                    root_data,
                    neighbour_data,
                    path,
                    cstep
                )

                state = self.sess.run(
                    self.model.net_rnn.final_state_encode,
                    feed_dict={
                        self.model.x_root: x_root,
                        self.model.x_neighbour: x_neight
                    })

                spred = decode_seq[:, 0:1, :]

                spredlist = list()
                for _ in range(config.out_seq_length):  # max sentence length
                    spred, state = self.sess.run([
                        self.model.test_net.outputs,
                        self.model.net_rnn.final_state_decode],
                        feed_dict={
                            self.model.net_rnn.initial_state_decode: state,
                            self.model.decode_seqs_test: spred
                        })
                    spredlist.append(spred)

                pred = np.concatenate(spredlist, axis=1)
                all_loss += np.mean(utils.mape(pred, target_seq[:, :-1, :]))
                time_loss += np.mean(utils.mape(pred[:, :config.out_seq_length, 0], target_seq[:, :config.out_seq_length, 0]), axis=0)

                predlist.append(pred)

            predlist = np.concatenate(predlist, axis=0)
            pathpred.append(predlist)

            if path % 500 == 0:
                print(
                    "[Test] Epoch: [%3d][%5d/%5d] time: %.4f, loss: %s, tloss: %s" %
                    (epoch, path, root_data.shape[0], time.time() - step_time, all_loss / round, time_loss / round)
                )

                tmppathpred = np.stack(pathpred, axis=0)
                savedir = config.result_path + self.model.model_name + "/"
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                np.savez(savedir + "%d_test" % (epoch), pred=tmppathpred)

            logger.add_log("%d_%s" % (epoch, pathlist[path]), list(all_loss / round) + list(time_loss / round))

        pathpred = np.stack(pathpred, axis=0)
        savedir = config.result_path + self.model.model_name + "/"
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        np.savez(savedir + "%d_test" % (epoch), pred=pathpred)

        print(
            "[Test Sum] Epoch [%3d]: time: %.4f, loss: %s, tloss: %s" %
            (epoch, time.time() - start_time, all_loss / round, time_loss / round)
        )

        return all_loss, time_loss, pathpred

    def controller_train(self, tepoch=config.epoch):
        root_data, neighbour_data, pathlist = dataloader.load_data(5, 5)

        last_save_epoch = self.base_epoch
        global_epoch = self.base_epoch + 1

        if last_save_epoch >= 0:
            self.restore_model(
                path=self.model_save_dir,
                global_step=last_save_epoch
            )

        logger_train = log.Logger(columns=["mae_copy", "loss", "nmse_train", "nmse", "rmse", "mae", "mape"])
        # logger_valid = log.Logger(columns=["mae_copy", "lossv", "nmse_test", "nmsev", "msev", "maev", "mapev"])
        # logger_test = log.Logger(columns=["mae_copy", "lossv", "nmse_test", "nmsev", "msev", "maev", "mapev"] + list(range(15, 121, 15)))
        logger_test = log.Logger(columns=["mapev"] + list(range(15, 121, 15)))

        for epoch in range(tepoch + 1):

            self.__train__(global_epoch, root_data[:, :-config.valid_length, :], neighbour_data[:, :-config.valid_length, :], logger_train)

            if epoch % config.test_p_epoch == 0:
                # self.__valid__(global_epoch, root_data[:, -config.valid_length:, :], neighbour_data[:, -config.valid_length:, :], logger_valid)
                self.__test__(global_epoch, root_data[:, -config.valid_length:, :], neighbour_data[:, -config.valid_length:, :], logger_test, pathlist)

            if global_epoch > self.base_epoch and global_epoch % config.save_p_epoch == 0:
                self.save_model(
                    path=self.model_save_dir,
                    global_step=global_epoch
                )
                last_save_epoch = global_epoch

            logger_train.save(self.log_save_dir + config.global_start_time + "_train.csv")
            # logger_valid.save(self.log_save_dir + config.global_start_time + "_valid.csv")
            logger_test.save(self.log_save_dir + config.global_start_time + "_test.csv")

            global_epoch += 1

    def controller_test(self):
        # root_data,pathlist  = dataloader.load_data_all()
        root_data, neighbour_data, pathlist  = dataloader.load_data(5, 5)

        last_save_epoch = self.base_epoch
        global_epoch = self.base_epoch + 1

        assert last_save_epoch >= 0
        self.restore_model(
            path=self.model_save_dir,
            global_step=last_save_epoch
        )

        # logger_test = log.Logger(columns=["mae_copy", "lossv", "nmse_test", "nmsev", "msev", "maev", "mapev"] + list(range(15, 121, 15)))
        logger_test = log.Logger(columns=["mapev"] + list(range(15, 121, 15)))

        self.__test__(global_epoch, root_data[:, -config.valid_length:, :], neighbour_data[:, -config.valid_length:, :], logger_test, pathlist, test_interval=1)

        logger_test.save(self.log_save_dir + config.global_start_time + "_test.csv")

class Seq2Seq_Controller(Controller):

    def __train__(self, epoch, root_data, logger):

        all_loss = np.zeros(7)

        start_time = time.time()
        step_time = time.time()
        train_steps = (len(root_data) - config.in_seq_length - config.out_seq_length + 1) // config.batch_size

        for cstep in range(train_steps):

            x_root, decode_seq, target_seq = dataloader.get_train_data(root_data, cstep * config.batch_size)

            global_step = cstep + epoch * train_steps

            results = self.sess.run([
                self.model.mae_copy,
                self.model.train_loss,
                self.model.nmse_train_loss,
                self.model.nmse_train_noend,
                self.model.rmse_train_noend,
                self.model.mae_train_noend,
                self.model.mape_train_noend,
                self.model.learning_rate,
                self.model.optim],
                feed_dict={
                    self.model.x_root: x_root,
                    self.model.decode_seqs: decode_seq,
                    self.model.target_seqs: target_seq,
                    self.model.global_step: global_step
                })

            all_loss += np.array(results[:-2])

            if cstep % 60 == 0 and cstep > 0:
                print(
                    "[Train] Epoch: [%3d][%4d/%4d] time: %.4f, lr: %.8f, loss: %s" %
                    (epoch, cstep, train_steps, time.time() - step_time, results[-2], all_loss / (cstep + 1))
                )
                step_time = time.time()
                logger.add_log(global_step, all_loss / (cstep + 1))

        print(
            "[Train Sum] Epoch: [%3d] time: %.4f, lr: %.8f, loss: %s" %
            (epoch, time.time() - start_time, results[-2], all_loss / train_steps)
        )
        logger.add_log(global_step, all_loss / train_steps)

        return all_loss

    def __valid__(self, epoch, root_data, logger):

        all_loss = np.zeros(7)

        start_time = time.time()
        step_time = time.time()
        valid_steps = (len(root_data) - config.in_seq_length - config.out_seq_length + 1) // config.batch_size

        for cstep in range(valid_steps):

            x_root, decode_seq, target_seq = dataloader.get_train_data(root_data, cstep * config.batch_size)

            global_step = cstep + epoch * valid_steps

            results = self.sess.run([
                self.model.mae_copy,
                self.model.test_loss,
                self.model.nmse_test_loss,
                self.model.nmse_test_noend,
                self.model.rmse_test_noend,
                self.model.mae_test_noend,
                self.model.mape_test_noend],
                feed_dict={
                    self.model.x_root: x_root,
                    self.model.decode_seqs: decode_seq,
                    self.model.target_seqs: target_seq,
                })

            all_loss += np.array(results[:7])

            if cstep % 100 == 0 and cstep > 0:
                print(
                    "[Valid] Epoch: [%3d][%4d/%4d] time: %.4f, loss: %s" %
                    (epoch, cstep, valid_steps, time.time() - step_time, all_loss / (cstep + 1))
                )
                step_time = time.time()

        print(
            "[Valid Sum] Epoch: [%3d] time: %.4f, loss: %s" %
            (epoch, time.time() - start_time, all_loss / valid_steps)
        )
        logger.add_log(epoch, all_loss / valid_steps)

        return all_loss

    def __test__(self, epoch, root_data, logger, pathlist, test_interval=10):

        each_num_seq = root_data.shape[1] - (config.in_seq_length + config.out_seq_length) + 1
        total_batch_size = root_data.shape[0] * each_num_seq
        train_order = list(range(total_batch_size))

        all_loss = np.zeros(1)
        time_loss = np.zeros(config.out_seq_length)

        start_time = time.time()
        step_time = time.time()

        round = 0
        pathpred = list()
        for path in range(0, root_data.shape[0], test_interval):
            predlist = list()
            step_time = time.time()
            for cstep in range(each_num_seq // config.batch_size):
                round += 1

                x_root, decode_seq, target_seq = dataloader.get_minibatch_4_test(
                    root_data,
                    path,
                    cstep
                )

                state = self.sess.run(
                    self.model.net_rnn.final_state_encode,
                    feed_dict={
                        self.model.x_root: x_root
                    })

                spred = decode_seq[:, 0:1, :]

                spredlist = list()
                for _ in range(config.out_seq_length):  # max sentence length
                    spred, state = self.sess.run([
                        self.model.test_net.outputs,
                        self.model.net_rnn.final_state_decode],
                        feed_dict={
                            self.model.net_rnn.initial_state_decode: state,
                            self.model.decode_seqs_test: spred
                        })
                    spredlist.append(spred)

                pred = np.concatenate(spredlist, axis=1)
                all_loss += np.mean(utils.mape(pred, target_seq[:, :-1, :]))
                time_loss += np.mean(utils.mape(pred[:, :config.out_seq_length, 0], target_seq[:, :config.out_seq_length, 0]), axis=0)

                predlist.append(pred)

            predlist = np.concatenate(predlist, axis=0)
            pathpred.append(predlist)

            if path % 500 == 0:
                print(
                    "[Test] Epoch: [%3d][%5d/%5d] time: %.4f, loss: %s, tloss: %s" %
                    (epoch, path, root_data.shape[0], time.time() - step_time, all_loss / round, time_loss / round)
                )

                tmppathpred = np.stack(pathpred, axis=0)
                savedir = config.result_path + self.model.model_name + "/"
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                np.savez(savedir + "%d_test" % (epoch), pred=tmppathpred)

            logger.add_log("%d_%s" % (epoch, pathlist[path]), list(all_loss / round) + list(time_loss / round))

        pathpred = np.stack(pathpred, axis=0)
        savedir = config.result_path + self.model.model_name + "/"
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        np.savez(savedir + "%d_test" % (epoch), pred=pathpred)


        print(
            "[Test Sum] Epoch [%3d]: time: %.4f, loss: %s, tloss: %s" %
            (epoch, time.time() - start_time, all_loss / round, time_loss / round)
        )

        return all_loss, time_loss, pathpred

    def controller_train(self, tepoch=config.epoch):
        train_data = np.genfromtxt('../data/800r_train_2.txt')
        val_data = np.genfromtxt('../data/800r_test.txt')

        last_save_epoch = self.base_epoch
        global_epoch = self.base_epoch + 1

        if last_save_epoch >= 0:
            self.restore_model(
                path=self.model_save_dir,
                global_step=last_save_epoch
            )
            # tl.files.save_npz_dict(self.model.train_net.all_params, name=self.model_save_dir + "%d.npz" % global_epoch, sess=self.sess)
            # return

        # logger_train = log.Logger(columns=["mae_copy", "loss", "nmse_train", "nmse", "mse", "mae", "mape", "lossv", "nmse_test", "nmsev", "msev", "maev", "mapev"])
        logger_train = log.Logger(columns=["mae_copy", "loss", "nmse_train", "nmse", "rmse", "mae", "mape"])
        logger_valid = log.Logger(columns=["mae_copy", "lossv", "nmse_test", "nmsev", "rmsev", "maev", "mapev"])
        # logger_test = log.Logger(columns=["mae_copy", "lossv", "nmse_test", "nmsev", "msev", "maev", "mapev"] + list(range(15, 121, 15)))
        # logger_test = log.Logger(columns=["mapev"] + list(range(15, 121, 15)))

        for epoch in range(tepoch + 1):

            self.__train__(global_epoch, train_data, logger_train)

            if epoch % config.test_p_epoch == 0:
                self.__valid__(global_epoch, val_data, logger_valid)
                # self.__test__(global_epoch, root_data[:, -config.valid_length:, :], logger_test, pathlist)

            if global_epoch > self.base_epoch and global_epoch % config.save_p_epoch == 0:
                self.save_model(
                    path=self.model_save_dir,
                    global_step=global_epoch
                )
                last_save_epoch = global_epoch

            logger_train.save(self.log_save_dir + config.global_start_time + "_train.csv")
            logger_valid.save(self.log_save_dir + config.global_start_time + "_valid.csv")
            # logger_test.save(self.log_save_dir + config.global_start_time + "_test.csv")

            global_epoch += 1

    def controller_test(self):
        # root_data,pathlist  = dataloader.load_data_all()
        root_data, neighbour_data, pathlist  = dataloader.load_data(5, 5)
        del neighbour_data
        # event_data = dataloader.load_event_data()

        last_save_epoch = self.base_epoch
        global_epoch = self.base_epoch + 1

        assert last_save_epoch >= 0
        self.restore_model(
            path=self.model_save_dir,
            global_step=last_save_epoch
        )

        # logger_test = log.Logger(columns=["mae_copy", "lossv", "nmse_test", "nmsev", "msev", "maev", "mapev"] + list(range(15, 121, 15)))
        logger_test = log.Logger(columns=["mapev"] + list(range(15, 121, 15)))

        self.__test__(global_epoch, root_data[:, -config.valid_length:, :], logger_test, pathlist, test_interval=1)
        # self.__test_event__(global_epoch, root_data[:, -config.valid_length:, :], event_data, logger_test, pathlist, test_interval=1)

        logger_test.save(self.log_save_dir + config.global_start_time + "_test.csv")


if __name__ == "__main__":
    '''

    with tf.Graph().as_default() as graph:
        tl.layers.clear_layers_name()
        mdl = model.Spacial_Model(
            model_name="spacial_model",
            start_learning_rate=0.001,
            decay_steps=2e4,
            decay_rate=0.5,
        )
        ctl = Controller(model=mdl, base_epoch=100)
        # ctl.controller_train()
        # ctl.controller_train(tepoch=2)
        ctl.controller_test()
        ctl.sess.close()

    '''

    with tf.Graph().as_default() as graph:
        tl.layers.clear_layers_name()
        mdl = model.Seq2Seq_Model(
            model_name="seq2seq_model",
            start_learning_rate=0.001,
            decay_steps=1400,
            decay_rate=0.9,
        )
        ctl = Seq2Seq_Controller(model=mdl, base_epoch=-1)
        ctl.controller_train()
        # ctl.controller_test()
        ctl.sess.close()
    exit()


