import time
from flcore.clients.clientbase import Client
from torchvision.transforms import transforms
import copy
import torch.nn.functional as F
import torch
import torch.nn as nn
from utils.lesam_utils import LESAM


class clientLESAMD(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)
        self.dual_vec = None
        self.global_update = None
        self.local_update = None

    def train(self):
        base_optimizer = torch.optim.SGD(self.model.parameters(),
                         lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        optimizer = LESAM(self.model.parameters(), base_optimizer, rho=self.rho)
        trainloader = self.load_train_data()
        self.model.train()
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])
        regular_params = param_to_vector(self.model).detach()
        for step in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                if self.dataset != "agnews":
                    x = transform_train(x)
                optimizer.first_step(self.global_update)
                output = self.model(x)
                loss = self.loss(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.second_step()
                # dyn
                local_params = param_to_vector(self.model)
                loss = self.beta / 2 * torch.norm(local_params - regular_params, 2)
                loss += self.beta*torch.dot(local_params, self.dual_vec)
                loss.backward()
                base_optimizer.step()
        # DYN
        local_params = param_to_vector(self.model).detach()
        self.local_update = local_params - regular_params
        self.dual_vec += self.local_update

def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append((param.reshape(-1)))
    return torch.cat(vec)


