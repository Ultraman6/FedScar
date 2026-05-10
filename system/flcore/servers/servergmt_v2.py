import time
from flcore.clients.clientgmt_v2 import clientGMTV2
from flcore.servers.serverbase import Server
from tqdm import tqdm
import torch
import os
import copy
from utils.mem_utils import get_gpu_memory_usage
import statistics


class FedGMTV2(Server):
    def __init__(self, args, hyperparams=None):
        super().__init__(args, hyperparams)
        self.set_clients(args, clientGMTV2)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        for client in self.clients:
            client.EMA = copy.deepcopy(args.model)
        init_par_list = param_to_vector(self.global_model)
        self.dual_variable_list = torch.zeros((self.num_clients, init_par_list.shape[0])).to(self.device)

    def train(self):
        for epoch in tqdm(range(1, self.global_rounds + 1), desc='server-training'):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            # loss_before,_ = self.sharpness()
            print()

            for client in self.selected_clients:
                client.learning_rate = self.learning_rate
                client.train()
                self.dual_variable_list[client.id] += client.local_update

            self._lr_scheduler_()
            self.receive_models()
            self.aggregate_parameters()
            # dyn
            global_para = param_to_vector(self.global_model)
            global_para += torch.mean(self.dual_variable_list, dim=0)
            vector_to_param(global_para, self.global_model)

            self.empty_cache()
            print(f"\n-------------Round number: {epoch}-------------")
            print("\nEvaluate global model")
            # loss_after,train_acc = self.sharpness()
            # self.loss_diff.append(loss_before-loss_after)

            self.evaluate()
            print('-'*25, 'This global round time cost', '-'*25, time.time() - s_t)
            if epoch%self.save_gap == 0:
                self.save_results(epoch)
            if epoch == self.global_rounds:
                self.specific_results['max_test_acc'] = round(max(self.test_acc),4)
                self.specific_results['avg_max_test_acc'] = round(sum(sorted(self.test_acc)[-50:])/50,4)
                self.specific_results['std_max_test_acc'] = round(statistics.stdev(sorted(self.test_acc)[-50:]),4)
                self.specific_results['avg_test_acc'] = round(sum(self.test_acc[-50:]) / 50,4)
                self.specific_results['std_test_acc'] = round(statistics.stdev(self.test_acc[-50:]),4)
                print(f"Max test acc:{self.specific_results['max_test_acc']}")
                print(f"Avg Max 50 acc:{self.specific_results['avg_max_test_acc']}, std: {self.specific_results['std_max_test_acc']}")
                print(f"Avg last 50 round acc:{self.specific_results['avg_test_acc']}, std: {self.specific_results['std_test_acc']}")
                self.save_specific_results()

    def send_models(self):
        for client in self.selected_clients:
            client.model = copy.deepcopy(self.global_model)
            client.EMA = copy.deepcopy(self.EMA_model)
            # To save gpu space, we use the dual_variable on server. Note that dual_variable on client also can be computed locally.
            client.dual_variable = self.dual_variable_list[client.id]

    def empty_cache(self):
        for client in self.selected_clients:
            del client.model, client.EMA, client.local_update
            client.model = None
            client.EMA = None
            client.local_update = None
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
