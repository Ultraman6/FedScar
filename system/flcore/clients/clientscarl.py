from flcore.clients.clientbase import Client
from torchvision.transforms import transforms
import torch
from utils.optim_utils import ERM, ESAM

class clientScarl(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)
        self.bias_vec = None
        self.global_bias_vec = None
        self.local_vec = None
        self.var = 0.0

    def train(self):
        base_optimizer = torch.optim.SGD(self.model.parameters(),lr=self.learning_rate,weight_decay=self.weight_decay,momentum=self.momentum)
        if self.rho == 0:
            optimizer = ERM(self.model.parameters(), base_optimizer)
        else:
            optimizer = ESAM(self.model.parameters(), base_optimizer, rho=self.rho, adaptive=False)
        trainloader = self.load_train_data()
        self.model.train()
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        origin_params = param_to_vector(self.model).detach()
        delta = self.global_bias_vec - self.bias_vec
        for step in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                if self.dataset != "agnews":
                    x = transform_train(x)
                optimizer.paras = [x, y, self.loss, self.model]
                optimizer.step()
                vec_add_to_grad(delta*self.tau, self.model)
                base_optimizer.step()

        regular_params = param_to_vector(self.model).detach()
        local_update = regular_params - origin_params
        self.local_vec = local_update

def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append((param.reshape(-1)))
    return torch.cat(vec)

def vector_to_param(vector, model):
    # vector ---> model parameters
    vector = vector.detach().clone()
    index = 0
    for param in model.parameters():
        param_size = param.numel()
        param.data = vector[index:index + param_size].view(param.shape)
        index += param_size


def vec_add_to_grad(vec, model):
    # vec ---> grad addition
    vec = vec.detach().clone()
    index = 0
    for param in model.parameters():
        param_size = param.numel()
        param.grad.data += vec[index:index + param_size].view(param.shape)
        index += param_size