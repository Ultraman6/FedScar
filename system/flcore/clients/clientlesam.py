import time
from flcore.clients.clientbase import Client
from torchvision.transforms import transforms
import copy
import torch.nn.functional as F
import torch
import torch.nn as nn
from utils.lesam_utils import LESAM


class clientLESAM(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)
        self.global_update = None

    def train(self):
        base_optimizer = torch.optim.SGD(self.model.parameters(),
                         lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        optimizer = LESAM(self.model.parameters(), base_optimizer, rho=self.rho)
        trainloader = self.load_train_data()
        self.model.train()
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])
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
                base_optimizer.step()

def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append((param.reshape(-1)))
    return torch.cat(vec)


