import numpy as np
import torch
import math
from matplotlib import pyplot as plt
from typing import Optional

torch.set_default_dtype(torch.float64)


# Distributions.

def dirac(location: float, mesh_space: torch.tensor) -> torch.tensor:
    initial_condition = torch.zeros(mesh_space.size(0) - 2)
    mid = torch.argmin(torch.abs(mesh_space[1:-1] - location)).item()
    initial_condition[mid] = 1/(mesh_space[mid + 2] - mesh_space[mid + 1])
    return initial_condition


def uniform(lower: float, upper: float, mesh_space: torch.tensor) -> torch.tensor:
    initial_condition = torch.zeros(mesh_space.size(0) - 2)
    idx_0 = torch.argmin(torch.abs(mesh_space[1:-1] - lower)).item()
    idx_1 = torch.argmin(torch.abs(mesh_space[1:-1] - upper)).item()
    initial_condition[idx_0:idx_1] = 1/(mesh_space[idx_1 - 1] - mesh_space[idx_0 - 1])
    return initial_condition


def mollifier(x: torch.tensor, mean: float = 0., support: float = 1):
    y = (x - mean)/support
    z = torch.zeros(y.size())
    idx = torch.logical_and(torch.lt(y, torch.tensor(1.)), torch.gt(y, torch.tensor(-1.)))
    mass = 0.4439938161680794
    z[idx] = torch.exp(-1/(1 - y[idx]**2))/(mass*support)

    return z


def normal(x: torch.tensor, mean: float = 0., variance: float = 1.) -> torch.tensor:
    return torch.exp(-(x - mean)**2/(2*variance))/(2*math.pi*variance)**(1/2)


def gamma(x: torch.tensor, shape: float, rate: float):
    y = torch.maximum(x, torch.tensor(0.))
    return rate**shape*y**(shape - 1)*torch.exp(-rate*y)/math.gamma(shape)


def return_mesh_space(endpoints: torch.tensor, num_points: torch.tensor) -> torch.tensor:
    mesh = []
    for i in range(0, endpoints.size(0) - 2):
        mesh += [torch.linspace(endpoints[i], endpoints[i + 1], num_points[i] + 2)[:-1]]
    mesh += [torch.linspace(endpoints[-2], endpoints[-1], num_points[-1] + 2)]

    return torch.cat(mesh)


def loss_function(x: torch.tensor) -> torch.tensor:
    return torch.ones(x.size())


def heatplot(density: torch.tensor, mesh_time: torch.tensor, mesh_space: torch.tensor,
             min_space: Optional[float] = None, max_space:  Optional[float] = None, tol: Optional[float] = None):
    density_0 = torch.maximum(density, torch.zeros(1))
    # if min_space is not None and max_space is not None:
    #     idx_0 = torch.argmin(torch.abs(min_space - torch.tensor(min_space))).item()
    #     idx_1 = torch.argmin(torch.abs(min_space - torch.tensor(max_space))).item() + 1
    #     mesh_x, mesh_t = np.meshgrid(mesh_time, mesh_space[idx_0:idx_1])
    #
    #
    # idx = torch.ge(density, torch.tensor(10**(-5)))
    # idx = idx.any(0)
    # idx_0 = idx.to(dtype=torch.float32).argmax()
    # idx_1 = idx.size(0) - idx.to(dtype=torch.float32).flip(0).argmax()
    # idx[idx_0:idx_1] = True
    #
    # mesh_x, mesh_t = np.meshgrid(mesh_time, mesh_space[idx_0 + 1:idx_1 + 1])
    # fig, ax = plt.subplots()
    # ax.pcolormesh(mesh_x, mesh_t, density[:, idx_0:idx_1].transpose(0, 1).detach(), cmap='plasma')
    #
    # density_max = density.max()
    # density_min = density.min()
    # c = ax.pcolormesh(mesh_x, mesh_t, density[:, idx_0:idx_1].transpose(0, 1).detach(), cmap='plasma',
    #                   vmin=density_min, vmax=density_max)
    # fig.colorbar(c, ax=ax)
    # plt.show()
    mesh_x, mesh_t = np.meshgrid(mesh_time, mesh_space[1:-1])
    fig, ax = plt.subplots()
    ax.pcolormesh(mesh_x, mesh_t, density_0.transpose(0, 1).detach(), cmap='plasma')

    density_max = density_0.max()
    density_min = density_0.min()
    c = ax.pcolormesh(mesh_x, mesh_t, density_0.transpose(0, 1).detach(), cmap='plasma', vmin=density_min,
                      vmax=density_max)
    fig.colorbar(c, ax=ax)

    return fig, ax


def integrate(values: torch.tensor, density: torch.tensor, mesh_space: torch.tensor) -> torch.tensor:
    if values.dim() == 2:
        return torch.einsum('ij, ij -> i', values, density*(mesh_space[2:] - mesh_space[1:-1]))
    else:
        return torch.matmul(density*(mesh_space[2:] - mesh_space[1:-1]), values)


def inv_relu(z: torch.tensor) -> torch.tensor:
    return torch.relu(-z)


if __name__ == "__main__":
    shape = 6.
    rate = 60.
    z = torch.linspace(0., .5, 1000)
    plt.plot(z, gamma(z, shape, rate))
    plt.show()

    # z = torch.linspace(-1, 1, 5)
    # print(torch.lt(z, torch.tensor(0.)).double().dtype)
    #
    # intensity = 10.
    # z = torch.linspace(0., .5, 1000)
    # plt.plot(z, intensity*z)
    # plt.plot(z, (intensity*z)**2)
    # plt.plot(z, torch.exp(intensity*z) - 1)
    # plt.show()
