import time
from flcore.servers.serverbase import Server
from tqdm import tqdm
import torch
import os
import copy
from utils.mem_utils import get_gpu_memory_usage
import statistics

import copy
import time
import torch
from flcore.clients.clientdyn import clientDyn
from flcore.servers.serverbase import Server
from threading import Thread

class FedDyn(Server):
    def __init__(self, args, hyperparams=None):
        super().__init__(args, hyperparams)
        self.set_clients(args, clientDyn)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        init_par_list = torch.zeros_like(param_to_vector(self.global_model))
        self.dual_vec_list = torch.zeros((self.num_clients, init_par_list.shape[0])).to(self.device)
        for client in self.clients:
            client.dual_vec = self.dual_vec_list[client.id]
            client.local_vec = copy.deepcopy(init_par_list)
        self.dual_vec = copy.deepcopy(init_par_list)

    def train(self):
        st = 0.0
        st_ = time.time()
        for epoch in tqdm(range(1, self.global_rounds+1), desc=f'server-training-{self.algorithm}'):
            self.selected_clients = self.select_clients()
            self.send_models()
            # loss_before,_ = self.sharpness()
            # print()
            s_t = time.time()
            for client in self.selected_clients:
                client.learning_rate = self.learning_rate
                client.train()
            st += time.time() - s_t
            self.time.append(st)
            self._lr_scheduler_()
            self.receive_models()
            self.aggregate_parameters()
            global_para = param_to_vector(self.global_model)
            self.dual_vec += torch.sum(torch.stack([c.local_vec for c in self.selected_clients]),
                                       dim=0)*(self.alpha/self.num_clients)
            vector_to_param(global_para + self.dual_vec/self.alpha, self.global_model)
            # print(f"\n-------------Round number: {epoch}-------------")
            # print("\nEvaluate global model")
            # self.flatness_discrepancy()
            # self.evaluate()
            # self.empty_cache()
            # print('-'*25, 'This global round time cost', '-'*25, time.time() - s_t)
            self.time_whole.append(time.time()-st_)
            if epoch%self.save_gap == 0:
                self.save_results(epoch)
            # if epoch == self.global_rounds:
            #     self.specific_results['max_test_acc'] = round(max(self.test_acc),4)
            #     self.specific_results['avg_max_test_acc'] = round(sum(sorted(self.test_acc)[-50:])/50,4)
            #     self.specific_results['std_max_test_acc'] = round(statistics.stdev(sorted(self.test_acc)[-50:]),4)
            #     self.specific_results['avg_test_acc'] = round(sum(self.test_acc[-50:]) / 50,4)
            #     self.specific_results['std_test_acc'] = round(statistics.stdev(self.test_acc[-50:]),4)
            #     print(f"Max test acc:{self.specific_results['max_test_acc']}")
            #     print(f"Avg Max 50 acc:{self.specific_results['avg_max_test_acc']}, std: {self.specific_results['std_max_test_acc']}")
            #     print(f"Avg last 50 round acc:{self.specific_results['avg_test_acc']}, std: {self.specific_results['std_test_acc']}")
            #     self.save_specific_results()

    def send_models(self):
        for client in self.selected_clients:
            client.model = copy.deepcopy(self.global_model)

    def empty_cache(self):
        for client in self.selected_clients:
            del client.model,  client.local_vec
            client.model = None
            client.local_vec = None

        allocated, reserved = get_gpu_memory_usage(self.device)
        print(f"allocated GPU space: {allocated:.2f} MB，reserved GPU space: {reserved:.2f} MB")

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

