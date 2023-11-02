import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import cm
from pso import PSO
from tc.tc_fc import TTLinear

x_grid = np.linspace(0, 1, 51)
t_grid = np.linspace(0, 1, 51)

x = torch.from_numpy(x_grid)
t = torch.from_numpy(t_grid)

grid = torch.cartesian_prod(x, t).float()


def nn_autograd_simple(model, points, order, axis=0):
    points.requires_grad = True
    f = model(points).sum()
    for i in range(order):
        grads, = torch.autograd.grad(f, points, create_graph=True)
        f = grads[:, axis].sum()
    return grads[:, axis]


func_bnd1 = lambda x: 10 ** 4 * np.sin((1 / 10) * x * (x - 1)) ** 2
bnd1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
bndval1 = func_bnd1(bnd1[:, 0])

# du/dx (x,0) = 1e3*sin^2(x(x-1)/10)
func_bnd2 = lambda x: 10 ** 3 * np.sin((1 / 10) * x * (x - 1)) ** 2
bnd2 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
bop2 = {
    'du/dt':
        {
            'coeff': 1,
            'du/dt': [1],
            'pow': 1,
            'var': 0
        }
}
bndval2 = func_bnd2(bnd2[:, 0])

# u(0,t) = u(1,t)
bnd3_left = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()
bnd3_right = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()
bnd3 = [bnd3_left, bnd3_right]

# du/dt(0,t) = du/dt(1,t)
bnd4_left = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()
bnd4_right = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()
bnd4 = [bnd4_left, bnd4_right]

bop4 = {
    'du/dx':
        {
            'coeff': 1,
            'du/dx': [0],
            'pow': 1,
            'var': 0
        }
}
bcond_type = 'periodic'

bconds = [[bnd1, bndval1, 'dirichlet'],
          [bnd2, bop2, bndval2, 'operator'],
          [bnd3, bcond_type],
          [bnd4, bop4, bcond_type]]


def wave_op(model, grid):
    u_xx = nn_autograd_simple(model, grid, order=2, axis=0)
    u_tt = nn_autograd_simple(model, grid, order=2, axis=1)
    a = -(1 / 4)

    op = u_tt + a * u_xx

    return op


def op_loss(operator):
    return torch.mean(torch.square(operator))


def bcs_loss(model):
    bc1 = model(bnd1)
    bc2 = nn_autograd_simple(model, bnd2, order=1, axis=1)
    bc3 = model(bnd3_left) - model(bnd3_right)
    bc4 = nn_autograd_simple(model, bnd4_left, order=1, axis=0) - nn_autograd_simple(model, bnd4_right, order=1, axis=0)

    loss_bc1 = torch.mean(torch.square(bc1.reshape(-1) - bndval1))
    loss_bc2 = torch.mean(torch.square(bc2.reshape(-1) - bndval2))
    loss_bc3 = torch.mean(torch.square(bc3))
    loss_bc4 = torch.mean(torch.square(bc4))

    loss = loss_bc1 + loss_bc2 + loss_bc3 + loss_bc4
    return loss

hid = [5, 2, 5, 2]
rank = [1, 3, 3, 3, 1]

model = torch.nn.Sequential(
        nn.Linear(2, 100),
        nn.Tanh(),
        TTLinear(hid, hid, rank, activation=None),
        nn.Tanh(),
        TTLinear(hid, hid, rank, activation=None),
        nn.Tanh(),
        nn.Linear(100, 1))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

def closure():
    optimizer.zero_grad()
    operator = wave_op(model, grid)
    loss = op_loss(operator) + 1000 * bcs_loss(model)
    loss.backward()
    return loss


def draw_fig(model, grid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = grid[:, 0].detach().numpy().reshape(-1)
    ys = grid[:, 1].detach().numpy().reshape(-1)
    zs = model(grid).detach().numpy().reshape(-1)

    ax.plot_trisurf(xs, ys, zs, cmap=cm.jet, linewidth=0.2, alpha=1)

    ax.set_title("wave periodic")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")

    plt.show()


pso = PSO(model,
          ics_loss=op_loss,
          ics_fn=wave_op,
          bcs_fn=bcs_loss,
          grid=grid,
          n_iter=300,
          pop_size=60,
          gd_alpha=1e-3,
          verbose=True)

for j in range(10):
    pso.train()
    pso.get_best()
    model = pso.model
    draw_fig(model,grid)

