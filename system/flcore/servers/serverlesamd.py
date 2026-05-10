import time
from flcore.clients.clientlesamd import clientLESAMD
from flcore.servers.serverbase import Server
from tqdm import tqdm
import torch
import os
import copy
from utils.mem_utils import get_gpu_memory_usage
import statistics

class FedLESAMD(Server):
    def __init__(self, args, hyperparams=None):
        super().__init__(args, hyperparams)
        self.set_clients(args, clientLESAMD)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        init_par_list = param_to_vector(self.global_model)
        self.dual_vec_list = torch.zeros((self.num_clients, init_par_list.shape[0])).to(self.device)
        for client in self.clients:
            client.dual_vec = self.dual_vec_list[client.id]
        self.global_update = torch.zeros_like(param_to_vector(args.model))

    def train(self):
        # st = 0.0
        # st_ = time.time()
        for epoch in tqdm(range(1, self.global_rounds+1), desc=f'server-training-{self.algorithm}'):
            self.selected_clients = self.select_clients()
            self.send_models()
            # loss_before = self.sharpness()
            # print()
            self.client_updates = []
            # s_t = time.time()
            for client in self.selected_clients:
                client.learning_rate = self.learning_rate
                client.train()
            # st+=time.time()-s_t
            # self.time.append(st)
            self._lr_scheduler_()
            self.receive_models()
            regular_params = param_to_vector(self.global_model)
            self.aggregate_parameters()
            # global_para = param_to_vector(self.global_model)
            # global_update = global_para - regular_params
            # variance = 0.0
            # for c in self.clients:
            #     if c.local_update is not None:
            #         variance += torch.norm(c.local_update - global_update, p=2).item()
            # variance /= self.num_clients
            # self.variance.append(variance)
            global_para = param_to_vector(self.global_model)
            global_para += torch.mean(self.dual_vec_list, dim=0)
            vector_to_param(global_para, self.global_model)
            self.global_update = get_params_list_with_shape(self.global_model, (regular_params - global_para))
            # print(f"\n-------------Round number: {epoch}-------------")
            # print("\nEvaluate global model")
            # print('-'*25, 'This global round time cost', '-'*25, time.time() - s_t)
            # self.loss_diff.append(abs(loss_before - self.sharpness()))
            # self.flatness_discrepancy()
            if epoch >= self.eval_round: self.evaluate()
            # self.empty_cache()
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
                # print(f"Global flatness:{self.specific_results['global_flatness']}")
                # print(f"Flatness discrepancy:{self.specific_results['flatness_discrepancy']}")
                self.save_specific_results()

    def send_models(self):
        for client in self.selected_clients:
            client.model = copy.deepcopy(self.global_model)
            client.global_update = self.global_update

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