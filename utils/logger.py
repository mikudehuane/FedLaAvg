import csv
import os
import os.path as osp
import pickle
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter

import common


class Logger:
    META_FN = "meta.pkl"
    MODEL_FN = "model.pkl"
    SERVER_FN = "server.pkl"
    CLIENTS_FN = "clients.pkl"

    criteria = ["test_accuracy", "test_loss", "train_accuracy", "train_loss", "test_grad_norm", "train_grad_norm",
                "gs_rel_error", "gs_error_norm", "test_auc", "train_auc"]

    def __init__(self, run_name):
        self.father_fd = osp.join(common.project_dir, common.record_fd, run_name)
        self.log_fd = osp.join(self.father_fd, common.log_fd)
        self.tb_fd = osp.join(self.father_fd, common.tensorboard_fd)
        self.ck_fd = osp.join(self.father_fd, common.checkpoint_fd)

        os.makedirs(self.tb_fd, exist_ok=True)
        os.makedirs(self.log_fd, exist_ok=True)
        os.makedirs(self.ck_fd, exist_ok=True)

        # make the soft link to tensorboard
        slink_fp = osp.join(common.tb_slink_fd, run_name)
        if not osp.exists(slink_fp):
            os.makedirs(common.tb_slink_fd, exist_ok=True)
            tb_abs_path = osp.abspath(self.tb_fd)
            os.symlink(tb_abs_path, slink_fp)

        if self.has_checkpoint():
            c_round = self.load_meta()['current_round']
        else:
            c_round = 0

        # remove dirty data
        for _fn in Logger.criteria:
            _fp = osp.join(self.log_fd, _fn + '.csv')
            try:
                with open(_fp, 'r') as f_in:
                    valid_content = [f_in.readline()]
                    for line in f_in.readlines():
                        if int(line.split(',')[1]) < c_round:
                            valid_content.append(line)
                        else:
                            break
                    valid_content = ''.join(valid_content)
                with open(_fp, 'w') as f_out:
                    f_out.write(valid_content)
            except FileNotFoundError:
                with open(_fp, 'w') as f_out:
                    f_out.write("Time,Step,Value\n")
        self.log_handlers = {_fn: open(osp.join(self.log_fd, _fn + '.csv'), 'a') for _fn in Logger.criteria}

        self.summaryWriter = SummaryWriter(log_dir=self.tb_fd, purge_step=c_round)

        self.__tmp_checkpoint_fd = None

    def backup_checkpoint(self):
        self.__tmp_checkpoint_fd = osp.join(osp.split(self.ck_fd)[0], osp.split(self.ck_fd)[1] + "_bak")
        if osp.exists(self.__tmp_checkpoint_fd):
            raise FileExistsError(f"{self.__tmp_checkpoint_fd} already exists, can not be a backup destination")
        shutil.copytree(self.ck_fd, self.__tmp_checkpoint_fd)

    def remove_backup(self):
        shutil.rmtree(self.__tmp_checkpoint_fd)
        self.__tmp_checkpoint_fd = None

    def has_checkpoint(self):
        return osp.exists(osp.join(self.ck_fd, Logger.META_FN))

    def load_meta(self):
        return pickle.load(open(osp.join(self.ck_fd, Logger.META_FN), 'rb'))

    def dump_meta(self, meta):
        pickle.dump(meta, open(osp.join(self.ck_fd, Logger.META_FN), 'wb'))

    def load_model(self, model):
        model.load_state_dict(torch.load(osp.join(self.ck_fd, Logger.MODEL_FN)))

    def dump_model(self, model, fn=None):
        if fn is None:
            fn = Logger.MODEL_FN
        torch.save(model.state_dict(), osp.join(self.ck_fd, fn))

    def load_server(self):
        return pickle.load(open(osp.join(self.ck_fd, Logger.SERVER_FN), 'rb'))

    def dump_server(self, server):
        pickle.dump(server, open(osp.join(self.ck_fd, Logger.SERVER_FN), 'wb'))

    def load_clients(self, clients, target_alg):
        if target_alg not in ('fedavg', 'lastavg'):
            raise ValueError(f"target_alg={target_alg} not recognized in load_clients")
        dumped_clients = pickle.load(open(osp.join(self.ck_fd, Logger.CLIENTS_FN), 'rb'))
        load_last_grad_sum = dict(fedavg=False, lastavg=True)[target_alg]
        for client, state_dict in zip(clients, dumped_clients):
            client.load_state_dict(state_dict, load_last_grad_sum=load_last_grad_sum)

    def dump_clients(self, clients):
        dumped_clients = []
        for client in clients:
            dumped_clients.append(client.state_dict())
        pickle.dump(dumped_clients, open(osp.join(self.ck_fd, Logger.CLIENTS_FN), 'wb'))

    def reserve_checkpoint(self):
        checkpoint_round = self.load_checkpoint_round()
        shutil.copytree(self.father_fd, osp.join(osp.split(self.father_fd)[0],
                                             osp.split(self.father_fd)[1] + f"_step{checkpoint_round}"))

    def load_checkpoint_round(self):
        if self.has_checkpoint():
            return self.load_meta()['current_round']
        else:
            return 0

    def add_scalar(self, tag, scalar, x, time='', write_file=True):
        self.summaryWriter.add_scalar(tag, scalar, x)
        if write_file:
            log_f = self.log_handlers[tag]
            log_f.write(f"{time},{x},{scalar}\n")

    def add_meta(self, pars, argv, current_round):
        with open(osp.join(self.log_fd, 'meta.txt'), 'a') as f:
            f.write(f"Training starts from {current_round}\n")
            command = 'python ' + ' '.join(argv)
            f.write(f"Command: {command}\n")
            f.write(str(pars) + '\n')
            f.write('-' * 80 + '\n\n')

    def add_error(self, msg):
        with open(osp.join(self.log_fd, 'error.txt'), 'a') as f:
            f.write(msg + '\n')

    def add_statistics(self, clients, c_round):
        print("log statistics")
        with open(osp.join(self.log_fd, 'statistics.txt'), 'w') as f:
            f.write(f"current_round: {c_round}\n")
            for client in clients:
                f.write(f"client: {client.id}, r: {client.r}, num_trained: {len(client.train_log)}\n")
                f.write(f"{client.train_log}\n\n")

    def close(self):
        self.summaryWriter.close()
        for _, _f in self.log_handlers.items():
            _f.close()

    def flush(self):
        self.summaryWriter.flush()
        for _, _f in self.log_handlers.items():
            _f.flush()


def _test():
    logger = Logger("debug")
    logger.add_scalar("toy", 100, 0, 0.0)
    logger.add_scalar("toy", 20, 1, 1.0)
    logger.add_scalar("toy", 50, 2, 3.0)


if __name__ == "__main__":
    _test()

