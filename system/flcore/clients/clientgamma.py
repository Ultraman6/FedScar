import time
from flcore.clients.clientbase import Client
from torchvision.transforms import transforms
from utils.optim_utils import SAM, param_to_vector, vector_to_param
import torch
import copy

class clientGamma(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)
        # Initialize local control variable c_i
        with torch.no_grad():
            # Will be initialized after model is set
            self.c_i = None
        self.c = None  # Global control variable received from server
        self.delta_c_i = None  # Will be computed after training
        self.local_vec = None

    def train(self):
        # Initialize c_i if not already initialized
        if self.c_i is None:
            with torch.no_grad():
                self.c_i = torch.zeros_like(param_to_vector(self.model)).to(self.device)

        # Ensure c is on the correct device
        if self.c is None:
            self.c = torch.zeros_like(param_to_vector(self.model)).to(self.device)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                    weight_decay=self.weight_decay, momentum=self.momentum)
        minimizer = SAM(optimizer, self.model, self.rho)
        trainloader = self.load_train_data()
        self.model.train()

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        # Store initial model parameters
        with torch.no_grad():
            initial_params = param_to_vector(self.model).clone()

        data_size = 0
        K = 0  # Count of gradient steps
        origin_params = param_to_vector(self.model).detach()
        for step in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)
                if self.dataset != "agnews":
                    x = transform_train(x)

                # Store original parameters before ascent step
                with torch.no_grad():
                    origin_param = param_to_vector(self.model).clone()

                K += 1

                # Ascent Step
                optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                minimizer.ascent_step()

                # Descent Step
                optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()

                # Get gradient
                g_hat = minimizer.get_model_gradients()

                # Compute adjusted gradient: g_hat - c_i + c
                optimizer.zero_grad()
                grad = g_hat - self.c_i + self.c

                # Restore original parameters
                with torch.no_grad():
                    vector_to_param(self.model, origin_param)
                # Set gradient manually
                vector_to_grad(self.model, grad)
                # Perform optimizer step
                optimizer.step()
                data_size += len(y)

        # Compute delta_c_i
        with torch.no_grad():
            final_params = param_to_vector(self.model)
            if K > 0:
                self.delta_c_i = (initial_params - final_params) / K - self.c
                # Update c_i
                self.c_i = self.c_i + self.delta_c_i
            else:
                self.delta_c_i = torch.zeros_like(self.c_i)

        regular_params = param_to_vector(self.model).detach()
        self.local_vec = regular_params - origin_params


def vector_to_grad(model, vec):
    # vector ---> model gradients (same storage)
    current_index = 0
    for param in model.parameters():
        numel = param.data.numel()
        size = param.data.size()
        if param.grad is None:
            param.grad = torch.zeros_like(param.data)
        param.grad.copy_(vec[current_index:current_index + numel].view(size))
        current_index += numel