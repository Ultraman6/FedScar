import time
from flcore.clients.clientlesams import clientLESAMS
from flcore.servers.serverbase import Server
from tqdm import tqdm
import torch
import os
import copy
from utils.mem_utils import get_gpu_memory_usage
import statistics

class FedLESAMS(Server):
    def __init__(self, args, hyperparams=None):
        super().__init__(args, hyperparams)
        # Initialize global control variable c
        with torch.no_grad():
            self.c = torch.zeros_like(self._model_to_vector(self.global_model)).to(self.device)
        self.set_clients(args, clientLESAMS)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.global_update = torch.zeros_like(param_to_vector(args.model))

    def _model_to_vector(self, model):
        """Convert model parameters to a flattened vector"""
        vec = []
        for param in model.parameters():
            vec.append(param.data.reshape(-1))
        return torch.cat(vec)

    def _vector_to_model(self, model, vector):
        """Set model parameters from a flattened vector"""
        current_index = 0
        for param in model.parameters():
            numel = param.data.numel()
            size = param.data.size()
            param.data.copy_(vector[current_index:current_index + numel].view(size))
            current_index += numel

    def send_models(self, selected_clients=None, model=None):
        """Send model and global control variable c to clients"""
        if selected_clients is None:
            selected_clients = self.selected_clients
        if model is None:
            model = self.global_model
        assert (len(selected_clients) > 0)

        for client in selected_clients:
            client.model = copy.deepcopy(model)
            client.c = self.c.clone().to(client.device)
            client.global_update = self.global_update

    def receive_models(self):
        """Receive model updates and delta_c from clients"""
        assert (len(self.selected_clients) > 0)
        receive_clients = self.selected_clients
        self.uploaded_ids = []
        self.uploaded_weights = []  # num of samples
        self.uploaded_models = []
        self.uploaded_delta_c = []  # delta_c from each client

        tot_samples = 0
        for client in receive_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model.state_dict())
            if hasattr(client, 'delta_c_i'):
                self.uploaded_delta_c.append(client.delta_c_i)
            else:
                # If client didn't return delta_c, create zero vector
                self.uploaded_delta_c.append(torch.zeros_like(self.c))

        # Normalize weights
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        """Aggregate model parameters and update global control variable c"""
        assert (len(self.uploaded_models) > 0)
        # 1. Model Aggregation (FedAvg) - 保持不变
        fedavg_global_params = self.global_model.state_dict()
        for name_param in self.uploaded_models[0]:
            list_values_param = []
            for dict_local_params, local_weight in zip(self.uploaded_models, self.uploaded_weights):
                list_values_param.append(dict_local_params[name_param] * local_weight)
            value_global_param = sum(list_values_param)
            fedavg_global_params[name_param] = value_global_param
        self.global_model.load_state_dict(fedavg_global_params)
        # 2. Control Variate Aggregation - 需要修正
        # 计算选中客户端的 Delta c 的加权平均 (代表本轮参与者的平均漂移)
        dc_avg = sum(delta_c * weight for delta_c, weight in zip(self.uploaded_delta_c, self.uploaded_weights))
        # 修正关键点：根据 SCAFFOLD/FedGAMMA 原理，更新步长需要乘以参与率 (Join Ratio)
        # 公式: c_new = c_old + (|S| / N) * Average_Delta_c
        # 这里 len(self.uploaded_models) 就是 |S|
        update_step = dc_avg * (len(self.uploaded_models) / self.num_clients)
        self.c = self.c + update_step.to(self.device)

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
            global_para = param_to_vector(self.global_model)
            self.global_update = get_params_list_with_shape(self.global_model, (regular_params - global_para))
            # global_update = global_para - regular_params
            # variance = 0.0
            # for c in self.clients:
            #     if c.local_vec is not None:
            #         variance += torch.norm(c.local_vec - global_update, p=2).item()
            # variance /= self.num_clients
            # self.variance.append(variance)
            # print(f"\n-------------Round number: {epoch}-------------")
            # print("\nEvaluate global model")
            # self.loss_diff.append(abs(loss_before - self.sharpness()))
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
    #     """Clear client models and control variables to save memory"""
    #     for client in self.selected_clients:
    #         if hasattr(client, 'model'):
    #             del client.model
    #         if hasattr(client, 'delta_c_i'):
    #             del client.delta_c_i
    #         client.model = None
    #         client.delta_c_i = None
    #
    #     allocated, reserved = get_gpu_memory_usage(self.device)
    #     print(f"allocated GPU space: {allocated:.2f} MB，reserved GPU space: {reserved:.2f} MB")

def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append((param.reshape(-1).detach()))
    return torch.cat(vec)

def get_params_list_with_shape(model, param_list):
    vec_with_shape = []
    idx = 0
    for param in model.parameters():
        length = param.numel()
        vec_with_shape.append(param_list[idx:idx + length].reshape(param.shape))
    return vec_with_shape