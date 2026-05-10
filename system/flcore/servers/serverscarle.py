import math
import time
from flcore.clients.clientscarle import clientScarle
from flcore.servers.serverbase import Server
from utils.mem_utils import get_gpu_memory_usage
from tqdm import tqdm
import torch
import os
import copy
import statistics
import numpy as np
import torch.nn.functional as F

class FedScarle(Server):
    def __init__(self, args, hyperparams=None):
        super().__init__(args, hyperparams)
        self.set_clients(args, clientScarle)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        init_par_list = param_to_vector(self.global_model)
        self.dual_vec_list = torch.zeros((self.num_clients, init_par_list.shape[0])).to(self.device)
        self.bias_vec_list = torch.zeros((self.num_clients, init_par_list.shape[0])).to(self.device)
        for client in self.clients:
            client.bias_vec = self.bias_vec_list[client.id]
            client.dual_vec = self.dual_vec_list[client.id]
        self.global_update = torch.zeros_like(param_to_vector(args.model))

    def send_models(self, selected_clients, model):
        assert (len(self.clients) > 0)
        for client in selected_clients:
            _model = copy.deepcopy(model)
            model_vec = param_to_vector(_model)
            if client.model is not None:  # bias sharpness
                client_model_vec = param_to_vector(client.model)
                vec = model_vec - client_model_vec
                client.bias_vec.add_(vec.mul(client.var / vec.norm(p=2)))
                vector_to_param(model_vec + client.bias_vec, _model)
            client.model = _model
            client.global_update = self.global_update

    def train(self):
        # st=0.0
        # st_ = time.time()
        for epoch in tqdm(range(1, self.global_rounds+1), desc=f'server-training-{self.algorithm}'):
            self.bias_vec_list.mul_(self.gama)
            self.selected_clients = self.select_clients()
            self.send_models(self.selected_clients, self.global_model)
            # loss_before = self.sharpness()
            # print()
            self.client_updates = []
            bias_vec_mean = torch.mean(self.bias_vec_list, dim=0)
            # s_t = time.time()
            for client in self.selected_clients:
                client.global_bias_vec = bias_vec_mean
                client.learning_rate = self.learning_rate
                client.train()
            # st += time.time() - s_t
            # self.time.append(st)
            self._lr_scheduler_()
            self.receive_models()
            regular_params = param_to_vector(self.global_model)
            self.aggregate_parameters()
            global_para = param_to_vector(self.global_model)
            # global_update = global_para - regular_params
            # variance = 0.0
            # for c in self.clients:
            #     if c.local_vec is not None:
            #         variance += torch.norm(c.local_vec - global_update, p=2).item()
            # variance *= self.alpha/self.num_clients
            # self.variance.append(variance)
            for c in self.selected_clients:
                c.var = self.alpha
            global_para += self.beta*torch.mean(self.dual_vec_list - self.bias_vec_list, dim=0)
            vector_to_param(global_para, self.global_model)
            self.global_update = get_params_list_with_shape(self.global_model, (regular_params - global_para))
            # print(f"\n-------------Round number: {epoch}-------------")
            # print("\nEvaluate global model")
            # self.loss_diff.append(abs(loss_before - self.flatness_discrepancy()))
            # self.flatness_discrepancy()
            if epoch >= self.eval_round: self.evaluate()
            # self.empty_cache()
            # print('-'*25, 'This global round time cost', '-'*25, time.time() - s_t)
            # self.time_whole.append(time.time()-st_)
            if epoch%self.save_gap == 0:
                self.save_results(epoch)
            if epoch == self.global_rounds:
                self.specific_results['max_test_acc'] = round(max(self.test_acc),4)
                self.specific_results['avg_max_test_acc'] = round(sum(sorted(self.test_acc)[-50:])/50,4)
                self.specific_results['std_max_test_acc'] = round(statistics.stdev(sorted(self.test_acc)[-50:]),4)
                self.specific_results['avg_test_acc'] = round(sum(self.test_acc[-50:]) / 50,4)
                self.specific_results['std_test_acc'] = round(statistics.stdev(self.test_acc[-50:]),4)
                # self.specific_results['global_flatness'] = min(self.loss_diff)
                # self.specific_results['flatness_discrepancy'] = min(self.flat_disc)
                # print(f"Max test acc:{self.specific_results['max_test_acc']}")
                # print(f"Avg Max 50 acc:{self.specific_results['avg_max_test_acc']}, std: {self.specific_results['std_max_test_acc']}")
                # print(f"Avg last 50 round acc:{self.specific_results['avg_test_acc']}, std: {self.specific_results['std_test_acc']}")
                # print(f"Global flatness:{self.specific_results['global_flatness']}")
                # print(f"Flatness discrepancy:{self.specific_results['flatness_discrepancy']}")
                self.save_specific_results()

    # def empty_cache(self):
    #     for client in self.selected_clients:
    #         del client.model,  client.local_update
    #         client.model = None
    #         client.local_update = None
    #
    #     allocated, reserved = get_gpu_memory_usage(self.device)
    #     print(f"allocated GPU space: {allocated:.2f} MB，reserved GPU space: {reserved:.2f} MB")


def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append((param.reshape(-1).detach()))
    return torch.cat(vec)


def vector_to_param(vector, model):
    # vector ---> model parameters
    vector = vector.detach().clone()
    index = 0
    for param in model.parameters():
        param_size = param.numel()

        param.data = vector[index:index + param_size].view(param.shape)

        index += param_size

def get_params_list_with_shape(model, param_list):
    vec_with_shape = []
    idx = 0
    for param in model.parameters():
        length = param.numel()
        vec_with_shape.append(param_list[idx:idx + length].reshape(param.shape))
    return vec_with_shape