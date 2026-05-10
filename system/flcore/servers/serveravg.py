import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from tqdm import tqdm
import torch
import os, statistics

class FedAvg(Server):
    def __init__(self, args, hyperparams=None):
        super().__init__(args, hyperparams)
        self.set_clients(args, clientAVG)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        st = 0.0
        st_ = time.time()
        for epoch in tqdm(range(1, self.global_rounds+1), desc=f'server-training-{self.algorithm}'):
            self.selected_clients = self.select_clients()
            self.send_models(self.selected_clients,self.global_model)
            # print()
            self.client_updates = []
            s_t = time.time()
            for client in self.selected_clients:
                client.learning_rate = self.learning_rate
                client.train()
            st+=time.time()-s_t
            self.time.append(st)
            self._lr_scheduler_()
            self.receive_models()
            # regular_params = param_to_vector(self.global_model)
            self.aggregate_parameters()
            # global_para = param_to_vector(self.global_model)
            # global_update = global_para - regular_params
            # variance = 0.0
            # for c in self.clients:
            #     if c.local_vec is not None:
            #         variance += torch.norm(c.local_vec - global_update, p=2).item()
            # variance /= self.num_clients
            # self.variance.append(variance)
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
                # self.specific_results['max_test_acc'] = round(max(self.test_acc),4)
                # self.specific_results['avg_max_test_acc'] = round(sum(sorted(self.test_acc)[-50:])/50,4)
                # self.specific_results['std_max_test_acc'] = round(statistics.stdev(sorted(self.test_acc)[-50:]),4)
                # self.specific_results['avg_test_acc'] = round(sum(self.test_acc[-50:]) / 50,4)
                # self.specific_results['std_test_acc'] = round(statistics.stdev(self.test_acc[-50:]),4)
                # print(f"Max test acc:{self.specific_results['max_test_acc']}")
                # print(f"Avg Max 50 acc:{self.specific_results['avg_max_test_acc']}, std: {self.specific_results['std_max_test_acc']}")
                # print(f"Avg last 50 round acc:{self.specific_results['avg_test_acc']}, std: {self.specific_results['std_test_acc']}")
                # self.save_specific_results()

def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append((param.reshape(-1).detach()))
    return torch.cat(vec)
