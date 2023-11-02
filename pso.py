from filecmp import clear_cache
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import datetime
from typing import Union
from torch.optim.lr_scheduler import ExponentialLR
from copy import copy
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def progress(percent=0, width=30, metric=None, metricValue=None):
    """Prints on screen the current progress of a process.

    Args:
        percent (int, optional): The current progress (in percentiles). Defaults to 0.
        width (int, optional): The width size of the progress bar. Defaults to 30.
        metric (float, optional): The metric used. Defaults to None.
        metricValue (str, optional): The unit name for the metric used. Defaults to None.
    """
    left = width * int(percent) // 100
    right = width - left
    if metric:
        print(
            "\r[",
            "#" * left,
            " " * right,
            "]",
            f" {percent:.0f}%  --  ",
            "Current ",
            metric,
            ": ",
            metricValue,
            sep="",
            end="",
            flush=True,
        )
    else:
        print(
            "\r[",
            "#" * left,
            " " * right,
            "]",
            f" {percent:.0f}%  --  ",
            sep="",
            end="",
            flush=True,
        )


class PSO():
    def __init__(
            self,
            model,
            ics_fn,
            ics_loss,
            bcs_fn,
            grid,
            n_iter=2000,
            pop_size=30,
            b=0.99,
            c1=8e-2,
            c2=5e-1,
            gd_alpha=5e-3,
            verbose=False,
            c_decrease=False,
            beta_1=0.99,
            beta_2=0.999,
            epsilon=1e-8
    ):
        """The Particle Swarm Optimizer class. Specially built to deal with tensorflow neural networks.

        Args:
            loss_op (function): The fitness function for PSO.
            layer_sizes (list): The layers sizes of the neural net.
            n_iter (int, optional): Number of PSO iterations. Defaults to 2000.
            pop_size (int, optional): Population of the PSO swarm. Defaults to 30.
            b (float, optional): Inertia of the particles. Defaults to 0.9.
            c1 (float, optional): The *p-best* coeficient. Defaults to 0.8.
            c2 (float, optional): The *g-best* coeficient. Defaults to 0.5.
            gd_alpha (float, optional): Learning rate for gradient descent. Defaults to 0.00, so there wouldn't have any gradient-based optimization.
            verbose (bool, optional): Shows info during the training . Defaults to False.
        """
        self.model = model
        self.ics_fn = ics_fn
        self.ics_loss = ics_loss
        self.bcs_fn = bcs_fn
        self.grid = grid
        self.pop_size = pop_size
        vec_shape = torch.nn.utils.parameters_to_vector(
            self.model.parameters()).shape
        self.vec_shape = list(vec_shape)[0]
        self.n_iter = n_iter
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.swarm = self.build_swarm()
        self.loss = torch.inf
        self.loss_swarm, self.grads_swarm = self.fitness_fn()

        self.p, self.f_p = copy(self.swarm).detach(), copy(self.loss_swarm).detach()

        self.loss_history = []
        self.g_best = self.p[torch.argmin(self.f_p)]
        self.v = self.start_velocities()
        self.verbose = verbose
        self.verbose_milestone = np.linspace(0, n_iter, 11).astype(int)
        self.c_decrease = c_decrease
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m1 = torch.zeros(self.pop_size, self.vec_shape)
        self.m2 = torch.zeros(self.pop_size, self.vec_shape)
        self.gd_alpha = (
                gd_alpha * np.sqrt(1 - self.beta_2) / (1 - self.beta_1)
        )
        self.name = "PSO-GD"
        self.device = 'cpu'


    def loss_fn(self):
        op = self.ics_fn(self.model, self.grid)
        return self.ics_loss(op) + 1000 * self.bcs_fn(self.model)
    def loss_grads(self):
        loss = self.loss_fn()
        grads = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)
        grads = tuple(map(lambda x: torch.zeros(1) if x is None else x, grads))
        print(len(grads))
        grads = torch.nn.utils.parameters_to_vector(grads)

        return loss, grads

    def build_swarm(self):
        """Creates the swarm following the selected initialization method.

        Args:
            initialization_method (str): it uses uniform initialization.

        Returns:
            tf.Tensor: The PSO swarm population. Each particle represents a neural network.
        """
        vector = parameters_to_vector(self.model.parameters())
        matrix = []
        for _ in range(self.pop_size):
            matrix.append(vector.reshape(1, -1))
        matrix = torch.cat(matrix)
        error = torch.FloatTensor(self.pop_size, self.vec_shape).uniform_(-1e-2, 1e-2)
        swarm = (matrix + error).clone().detach().requires_grad_(True)
        return swarm

    def update_pso_params(self):
        self.c1 = self.c1 - 2 * self.c1 / self.n_iter
        self.c2 = self.c2 + self.c2 / self.n_iter

    def start_velocities(self):
        """Start the velocities of each particle in the population (swarm) as `0`.

        Returns:
            tf.Tensor: The starting velocities.
        """
        return torch.zeros((self.pop_size, self.vec_shape))

    def fitness_fn(self):
        """Fitness function for the whole swarm.

        Args:
            x (tf.Tensor): The swarm. All the particle's current positions. Which means the weights of all neural networks.

        Returns:
            tuple: the losses and gradients for all particles.
        """
        loss_swarm = []
        grads_swarm = []
        for particle in self.swarm:
            vector_to_parameters(particle, self.model.parameters())
            loss_particle, grads = self.loss_grads()
            loss_swarm.append(loss_particle)
            grads_swarm.append(grads.reshape(1, -1))

        return torch.stack(loss_swarm).reshape(-1), torch.vstack(grads_swarm)

    def get_randoms(self):
        """Generate random values to update the particles' positions.

        Returns:
            _type_: tf.Tensor
        """
        return torch.rand((2, 1, self.vec_shape))

    def update_p_best(self):
        """Updates the *p-best* positions."""

        idx = torch.where(self.loss_swarm < self.f_p)

        self.p[idx] = self.swarm[idx]
        self.f_p[idx] = self.loss_swarm[idx]

    def update_g_best(self):
        """Update the *g-best* position."""
        self.g_best = self.p[torch.argmin(self.f_p)]

    def gradient_descent(self):
        self.m1 = self.beta_1 * self.m1 + (1 - self.beta_1) * self.grads_swarm
        self.m2 = self.beta_2 * self.m2 + (1 - self.beta_2) * torch.square(
            self.grads_swarm)
        return self.gd_alpha * self.m1 / torch.sqrt(self.m2) + self.epsilon

    def step(self):
        """It runs ONE step on the particle swarm optimization."""
        r1, r2 = self.get_randoms()

        self.v = self.b * self.v + (1 - self.b) * (
                self.c1 * r1 * (self.p - self.swarm) + self.c2 * r2 * (self.g_best - self.swarm)
        )
        self.swarm = self.swarm + self.v - self.gradient_descent()
        self.loss_swarm, self.grads_swarm = self.fitness_fn()
        self.update_p_best()
        self.update_g_best()

    def train(self):
        """The particle swarm optimization. The PSO will optimize the weights according to the losses of the neural network, so this process is actually the neural network training."""
        for i in range(self.n_iter):
            self.step()
            self.loss_history.append(torch.min(self.loss_swarm).detach().cpu().numpy())

            if self.c_decrease:
                self.update_pso_params()
            if self.verbose and i in self.verbose_milestone:
                progress(
                    (i / self.n_iter) * 100,
                    metric="loss",
                    metricValue=self.loss_history[-1],
                )
        if self.verbose:
            progress(100)
            print()

    def get_best(self):
        """Return the *g-best*, the particle with best results after the training.

        Returns:
            tf.Tensor: the best particle of the swarm.
        """
        vector_to_parameters(self.g_best, self.model.parameters())

    def get_swarm_sln(self):
        """Return the swarm.

        Returns:
            tf.Tensor: The positions of each particle.
        """
        sln = []
        for particle in self.swarm:
            vector_to_parameters(particle, self.model.parameters())
            sln.append(self.model.forward(self.model.x_res).reshape(-1))
        return torch.stack(sln)

    def set_n_iter(self, n_iter):
        """Set the number of iterations.
        Args:
            x (int): Number of iterations.
        """
        self.n_iter = n_iter
        self.verbose_milestone = np.linspace(0, n_iter, 11).astype(int)