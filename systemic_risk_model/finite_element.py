import torch
import logging
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from typing import Callable, Tuple, Optional, Union, List

from systemic_risk_model import controls as ct
from systemic_risk_model import utilities as ut

torch.set_default_dtype(torch.float64)

class Solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, diagonals: torch.tensor, rhs: torch.tensor):
        diagonals_tf = tf.convert_to_tensor(diagonals.numpy())
        rhs_tf = tf.convert_to_tensor(rhs.numpy())
        solution = torch.tensor(tf.linalg.tridiagonal_solve(diagonals_tf, rhs_tf).numpy())
        ctx.save_for_backward(diagonals, solution)
        return solution

    @staticmethod
    def backward(ctx, grad_output):
        diagonals, solution = ctx.saved_tensors
        diagonals_tf = torch.zeros(diagonals.size())
        diagonals_tf[..., 0, :-1] = diagonals[..., 2, 1:]
        diagonals_tf[..., 1, :] = diagonals[..., 1, :]
        diagonals_tf[..., 2, 1:] = diagonals[..., 0, :-1]
        diagonals_tf = tf.convert_to_tensor(diagonals_tf.numpy())
        grad_output_tf = tf.convert_to_tensor(grad_output.numpy())
        gradient_rhs = torch.tensor(tf.linalg.tridiagonal_solve(diagonals_tf, grad_output_tf).numpy())

        gradient_diagonals = torch.zeros(diagonals.size())
        gradient_diagonals[..., 0, :-1] = -gradient_rhs[..., :-1]*solution[..., 1:]
        gradient_diagonals[..., 1, :] = -gradient_rhs[..., :]*solution[..., :]
        gradient_diagonals[..., 2, 1:] = -gradient_rhs[..., 1:]*solution[..., :-1]

        return gradient_diagonals, gradient_rhs


solve = Solve.apply


def get_intensity(intensity_parameter: float, intensity_type: str) -> Callable:
    if intensity_type == 'linear':
        return lambda x: intensity_parameter*torch.maximum(-x, torch.tensor(0.))
    elif intensity_type == 'quadratic':
        return lambda x: torch.maximum(-intensity_parameter*x, torch.tensor(0.))**2
    elif intensity_type == 'exponential':
        return lambda x: torch.exp(torch.maximum(-intensity_parameter*x, torch.tensor(0.))) - 1
    elif intensity_type == 'step':
        return lambda x: intensity_parameter*torch.lt(x, torch.tensor(0)).double()
    else:
        ValueError(f'Intensity type {intensity_type} is not admissible.')


class FinElteScheme(torch.nn.Module):
    def __init__(self, control: ct.Control, running_cost: Callable, terminal_cost: Callable):
        super().__init__()
        self.volatility = None
        self.volatility_0 = None
        self.diffusivity = None
        self.feedback = None
        self.intensity = None
        self.intensity_parameter = None
        self.intensity_type = None
        self.mean_reversion = None
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost

        self.control = control

    def initialise_parameters(self, volatility: float, volatility_0: float, feedback: float,
                              intensity_parameter: float, intensity_type: str = 'linear', mean_reversion: float = 0.):
        self.volatility = volatility
        self.volatility_0 = volatility_0
        self.diffusivity = (volatility**2 + volatility_0**2)/2
        self.feedback = feedback

        self.intensity_parameter = intensity_parameter
        self.intensity_type = intensity_type
        self.intensity = get_intensity(intensity_parameter, intensity_type)

        self.mean_reversion = mean_reversion

    def absorption(self, mesh_space: torch.tensor) -> torch.tensor:
        # Compute absorption diagonals.
        absorption = torch.zeros(3, mesh_space[1:-1].size(0))
        absorption[0, :-1] = (self.intensity(mesh_space[1:-2]) + self.intensity(mesh_space[2:-1]))/2
        absorption[1, :] = (self.intensity(mesh_space[:-2]) + 2*self.intensity(mesh_space[1:-1])
                            + self.intensity(mesh_space[2:]))
        absorption[0, 1:] = (self.intensity(mesh_space[1:-2]) + self.intensity(mesh_space[2:-1]))/2

        absorption = (mesh_space[1] - mesh_space[0])*absorption/6

        return absorption

    def drift(self, mesh_space: torch.tensor, density: torch.tensor, control: torch.tensor) -> torch.tensor:
        # Compute feedback term and drift coefficient.
        # feedback = self.feedback*ut.integrate(self.intensity(mesh_space[1:-1]).unsqueeze(0), density, mesh_space)
        feedback = (self.feedback*ut.integrate(self.intensity(mesh_space[1:-1]).unsqueeze(0), density, mesh_space)
                    + 0.5*ut.integrate(self.intensity(mesh_space[1:-1]).unsqueeze(0), density, mesh_space)**2)
        mean = ut.integrate(mesh_space[1:-1].unsqueeze(0), density, mesh_space)

        coeff = torch.zeros(feedback.size(0), control.size(-1) + 1)
        coeff[:, :-1] = -(control - feedback.unsqueeze(-1)
                          + self.mean_reversion*(mean.unsqueeze(-1) - mesh_space[1:-1]))
        coeff[:, -1] = -(control[..., -1] - feedback + self.mean_reversion*(mean - mesh_space[-1]))

        # Compute drift diagonals.
        drift = torch.zeros(density.size(0), 3, density.size(1))
        drift[:, 0, :-1] = -coeff[:, 1:-1]/2
        drift[:, 1, :] = (coeff[:, :-1] - coeff[:, 1:])/2
        drift[:, 2, 1:] = coeff[:, 1:-1]/2
        return drift

    def diffusion(self, mesh_space: torch.tensor) -> torch.tensor:
        # Compute diffusion matrix.
        diffusion = torch.zeros(3, mesh_space[1:-1].size(0))
        diffusion[0, :-1] = -1.
        diffusion[1, :] = 2.
        diffusion[2, 1:] = -1.
        diffusion = self.diffusivity*diffusion/(mesh_space[1] - mesh_space[0])
        return diffusion

    def stochastic_integral(self, density: torch.tensor, brownian_increment: torch.tensor, method: str = 'milstein'):
        density_0 = torch.zeros(density.size(0), density.size(1) + 2)
        density_0[:, 1:-1] = density
        return (self.volatility_0*brownian_increment/2).unsqueeze(-1)*(density_0[:, :-2] - density_0[:, 2:])

    def forward(self, mesh_time: torch.tensor, mesh_space: torch.tensor, initial_condition: torch.tensor,
                common_noise: Union[int, torch.tensor], method: str = 'milstein',
                trajectories: bool = True
                ) -> Union[Tuple[torch.tensor, torch.tensor, torch.tensor], Tuple[torch.tensor, torch.tensor]]:
        """
        Numerically computes a discretisation of the solution to one-dimensional Fokker-Planck equation with controlled
        drift, feedback, and absorption as well as the associated cost. The discretisastion is on mesh_time and
        mesh_space and has initial condition initial_condition.

        :param mesh_time: mesh for time discretisation; (num_t, )
        :param mesh_space: mesh for space discretisation; (num_x, )
        :param initial_condition: initial condition of pde; (num_x - 2, ) or (num_samples, num_x - 2)
        :param common_noise: if torch.tensor then they represent increments of common Brownian motion, if int
        then it represents number of samples to be generated for common Brownian motion, if None then method
        assumes no common Brownian motion: (num_samples, num_t) or int or None
        :param method: SDE discretisation scheme; 'milstein' or 'euler'
        :param trajectories: if trajectories = True keep entire trajectories of density, otherwise only density at
        final time
        :return: density, cost: solution of pde and associated cost; (num_t, num_x - 2), (0)
        """
        # Sample increments of Brownian motion if no increments are given, otherwise extract number of samples from
        # Brownian increments.
        if type(common_noise) == int:
            num_samples = common_noise
            brownian_increment = (mesh_time[1:] - mesh_time[:-1])**(1/2)*torch.randn(num_samples, mesh_time.size(0) - 1)
        else:
            num_samples = common_noise.size(0)
            brownian_increment = common_noise

        # Initialise density and cost tensors.
        if trajectories:
            density = torch.zeros(num_samples, mesh_time.size(0), initial_condition.size(-1))
            density[:, 0, :] = initial_condition
        else:
            density = torch.zeros(num_samples, initial_condition.size(-1))
            density[:, :] = initial_condition

        cost = torch.zeros(mesh_time.size(0))
        # TODO: change back.
        # self.control.initialise()

        for i in range(0, mesh_time.size(0) - 1):
            time_0 = mesh_time[i]
            time_1 = mesh_time[i + 1]
            if trajectories:
                density_0 = density[:, i, :]
            else:
                density_0 = density

            # TODO: change back.
            # Compute control and running cost.
            control = self.control(time_0, mesh_space, density_0)
            # control = self.control(time_0, mesh_space, time_1 - time_0, brownian_increment[:, i])
            cost[i] = (time_1 - time_0)*ut.integrate(self.running_cost(control), density_0, mesh_space).mean(0)

            # Compute absorption, drift, and diffusion matrix, and compute system matrix
            absorption = self.absorption(mesh_space)
            drift = self.drift(mesh_space, density_0, control)
            diffusion = self.diffusion(mesh_space)
            mass = torch.zeros(3, density.size(-1))
            mass[0, :-1] = 1.
            mass[1, :] = 4.
            mass[2, 1:] = 1.
            mass = (mesh_space[1] - mesh_space[0])*mass/6
            diagonals = mass + (time_1 - time_0)*(absorption + drift + diffusion)

            # Compute the increment stochastic integral and solve for next time step.
            stochastic_integral = self.stochastic_integral(density_0, brownian_increment[:, i], method=method)
            density_1 = torch.zeros(density_0.size(0), density_0.size(1) + 2)
            density_1[:, 1:-1] = density_0
            rhs = ((mesh_space[1] - mesh_space[0])*(density_1[:, 2:] + 4*density_1[:, 1:-1] + density_1[:, :-2])/6
                   + stochastic_integral)
            # rhs = (mesh_space[1] - mesh_space[0])*(density_1[:, 2:] + 4*density_1[:, 1:-1] + density_1[:, :-2])/6
            density_0 = solve(diagonals, rhs)

            if trajectories:
                density[:, i + 1, :] = density_0
            else:
                density = density_0

        # Compute terminal cost.
        if trajectories:
            cost[-1] = ut.integrate(self.terminal_cost(mesh_space[1:-1]).unsqueeze(0),
                                    density[:, -1, :], mesh_space).mean(0)
        else:
            cost[-1] = ut.integrate(self.terminal_cost(mesh_space[1:-1]).unsqueeze(0), density, mesh_space).mean(0)

        cost = torch.concat([cost[:-1].sum().unsqueeze(dim=0), cost[-1].unsqueeze(dim=0)], dim=0)

        return density, cost, brownian_increment

    def fit(self, mesh_time: torch.tensor, mesh_space: torch.tensor, initial_condition: torch.tensor, num_samples: int,
            num_epochs: int, lr: float = 0.01, print_frequency: int = 1, plot: bool = False):
        """
        Trains the FinDiffScheme fin_diff_scheme for num_epochs with learning rate lr.

        :param mesh_time: mesh for time discretisation in fin_diff_scheme; (num_t, )
        :param mesh_space: mesh for space discretisation in fin_diff_scheme; (num_x, )
        :param initial_condition: initial condition of pde; (num_x - 2, ) or (num_samples, num_x - 2)
        :param num_samples: number of Brownian paths; int
        :param num_epochs: number of epochs; int
        :param lr: learning rate; float
        :param print_frequency: number of epochs between print statements containing cost; int
        :param plot: if True plot loss and heatmap of density every frequency epoch; otherwise no plots
        :return: costs: tensor with costs during given epoch; (num_epochs, )
        """

        costs = torch.zeros(num_epochs, requires_grad=False)
        optimiser = torch.optim.Adam(params=self.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimiser, factor=0.1, patience=10)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser, T_max=num_epochs, eta_min=lr/100)
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimiser, total_iters=num_epochs)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimiser, step_size=100, gamma=0.1)

        for i in range(num_epochs):
            # Solve SDE from 0 to final_time.
            density, cost, _ = self.forward(
                mesh_time, mesh_space, initial_condition, common_noise=num_samples, trajectories=plot)
            cost = cost[0] + cost[1]

            optimiser.zero_grad()
            cost.backward()
            optimiser.step()

            costs[i] = cost.detach()

            if (i + 1) % print_frequency == 0:
                current_lr = [group['lr'] for group in optimiser.param_groups][0]
                cost_avg = costs[i + 1 - print_frequency:i + 1].mean()
                logging.info(f"Epoch {i + 1}: Cost = {cost_avg:.5}, Learning Rate = {current_lr}")

                if plot:
                    loss = 1 - (density*(mesh_space[2:] - mesh_space[1:-1])).sum(-1).mean(0)
                    plt.plot(mesh_time, loss.detach())
                    plt.show()

                    fig, ax = ut.heatplot(density[0, :, :], mesh_time, mesh_space)
                    fig.show()

            # scheduler.step()

        return costs, [group['lr'] for group in optimiser.param_groups][0]

    def clone(self, mesh_time: torch.tensor, mesh_space: torch.tensor, initial_condition: torch.tensor,
              control: Callable, num_samples: int, num_iters: int, num_epochs: int, batch_size: int, lr: float = 0.001,
              print_frequency: int = 10, plot: bool = False):
        costs = torch.zeros(num_iters, num_epochs, requires_grad=False)
        optimiser = torch.optim.Adam(params=self.parameters(), lr=lr, weight_decay=10.)

        for i in range(num_iters):
            # TODO: change back.
            # self.control.initialise()

            # Solve SDE from 0 to final_time.
            densities, _, _ = self.forward(mesh_time, mesh_space, initial_condition, common_noise=num_samples,
                                           trajectories=True)
            times = torch.tile(mesh_time, (num_samples, 1)).reshape(num_samples*mesh_time.size(0))
            dataset = torch.utils.data.TensorDataset(
                densities.reshape(num_samples*mesh_time.size(0), mesh_space.size(0) - 2).detach(), times)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

            for j in range(num_epochs):
                # TODO: change back.
                # self.control.initialise()
                loss_mean = torch.tensor(0.)
                for density, time in dataloader:
                    # TODO: change back.
                    loss = ((control(mesh_space[1:-1])
                             - self.control(time, mesh_space, density.detach())).norm(p=2, dim=None))**2
                    # loss = ((control(mesh_space[1:-1])
                    #          - self.control.readout(time, mesh_space)).norm(p=2, dim=None))**2
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    loss_mean = loss_mean + loss
                costs[i, j] = loss_mean/(num_samples*mesh_time.size(0))

                if (j + 1) % print_frequency == 0:
                    logging.info(f'Iteration {i + 1}, Epoch {j + 1}: Loss = {costs[i, j]}')

                    if plot:
                        # Plot control and density.
                        values = torch.zeros(mesh_time.size(0), mesh_space.size(0) - 2)
                        for t in range(mesh_time.size(0)):
                            # TODO: change back.
                            values[t, :] = self.control(mesh_time[t], mesh_space, densities[:1, t, :]).detach()
                            # values[t, :] = self.control.readout(mesh_time[t], mesh_space)[0, :].detach()

                        mesh_x, mesh_t = np.meshgrid(mesh_time, mesh_space[1:-1])
                        fig, ax = plt.subplots()
                        ax.pcolormesh(mesh_x, mesh_t, values.transpose(0, 1), cmap='plasma')

                        c = ax.pcolormesh(mesh_x, mesh_t, values.transpose(0, 1), cmap='plasma')
                        fig.colorbar(c, ax=ax)
                        plt.show()

                        # mesh_x, mesh_t = np.meshgrid(mesh_time, mesh_space[1:-1])
                        # fig, ax = plt.subplots()
                        # ax.pcolormesh(mesh_x, mesh_t, densities[0, :, :].detach().transpose(0, 1), cmap='plasma')
                        #
                        # c = ax.pcolormesh(mesh_x, mesh_t, densities[0, :, :].detach().transpose(0, 1), cmap='plasma')
                        # fig.colorbar(c, ax=ax)
                        # plt.show()

    def multilevel_fit(self, meshs_time: List, meshs_space: List, initial_conditions: List, num_samples: List,
                       num_epochs: List, lr: float = 0.001, print_frequency: int = 10, plot: bool = False):
        costs = torch.tensor([], requires_grad=False)
        for i in range(len(meshs_time)):
            logging.info(f"Training for {num_epochs[i]} epoch/s with time steps = {meshs_time[i].size(0)}, "
                         f"space steps = {meshs_space[i].size(0)}, samples = {num_samples[i]}.")
            costs_i, lr = self.fit(meshs_time[i], meshs_space[i], initial_conditions[i], num_samples[i], num_epochs[i],
                                   lr=lr, print_frequency=print_frequency, plot=plot)
            costs = torch.cat((costs, costs_i))

        return costs

    def get_state(self):
        state = {
            "volatility": self.volatility,
            "volatility_0": self.volatility_0,
            "feedback": self.feedback,
            "intensity_parameter": self.intensity_parameter,
            "intensity_type": self.intensity_type,
            "mean_reversion": self.mean_reversion,
            "control": self.control.state_dict()
        }
        return state

    @classmethod
    def load_state(cls, state, control, running_cost, terminal_cost):
        control.load_state_dict(state["control"])
        fin_elte_scheme = cls(control, running_cost, terminal_cost)
        fin_elte_scheme.initialise_parameters(state["volatility"], state["volatility_0"], state["feedback"],
                                              state["intensity_parameter"], state["intensity_type"])
        return fin_elte_scheme


class FinDiffScheme(torch.nn.Module):
    def __init__(self, control: ct.Control, running_cost: Callable, terminal_cost: Callable):
        super().__init__()
        self.volatility = None
        self.volatility_0 = None
        self.diffusivity = None
        self.feedback = None
        self.intensity = None
        self.intensity_parameter = None
        self.intensity_type = None
        self.mean_reversion = None
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost

        self.control = control

    def initialise_parameters(self, volatility: float, volatility_0: float, feedback: float,
                              intensity_parameter: float, intensity_type: str = 'linear', mean_reversion: float = 0.):
        self.volatility = volatility
        self.volatility_0 = volatility_0
        self.diffusivity = volatility**2 + volatility_0**2
        self.feedback = feedback

        self.intensity_parameter = intensity_parameter
        self.intensity_type = intensity_type
        self.intensity = get_intensity(intensity_parameter, intensity_type)

        self.mean_reversion = mean_reversion

    def absorption(self, time_0: torch.tensor, time_1: torch.tensor, mesh_space: torch.tensor) -> torch.tensor:
        # Compute absorption diagonals.
        absorption = torch.zeros(3, mesh_space[1:-1].size(0))
        absorption[1, :] = -(time_1 - time_0)*self.intensity(mesh_space[1:-1])
        return absorption

    def drift(self, time_0: torch.tensor, time_1: torch.tensor, mesh_space: torch.tensor, density: torch.tensor,
              control: torch.tensor) -> torch.tensor:
        # Compute feedback term and drift coefficient.
        feedback = self.feedback*ut.integrate(self.intensity(mesh_space[1:-1]).unsqueeze(0), density, mesh_space)
        mean = ut.integrate(mesh_space[1:-1].unsqueeze(0), density, mesh_space)
        coeff = -control + feedback.unsqueeze(-1) - self.mean_reversion*(mean.unsqueeze(-1) - mesh_space[1:-1])

        fwd = (time_1 - time_0)/(2*(mesh_space[2:] - mesh_space[1:-1]))
        bwd = (time_1 - time_0)/(2*(mesh_space[1:-1] - mesh_space[:-2]))

        # Compute drift diagonals.
        drift = torch.zeros(density.size(0), 3, density.size(1))
        drift[:, 0, :-1] = fwd[:-1]*coeff[:, 1:]
        drift[:, 1, :] = (bwd - fwd)*coeff
        drift[:, 2, 1:] = -bwd[1:]*coeff[:, :-1]
        return drift

    def diffusion(self, time_0: torch.tensor, time_1: torch.tensor, mesh_space: torch.tensor) -> torch.tensor:
        ctl = mesh_space[2:] - mesh_space[:-2]
        fwd = (self.diffusivity*(time_1 - time_0)/ctl)/(mesh_space[2:] - mesh_space[1:-1])
        bwd = (self.diffusivity*(time_1 - time_0)/ctl)/(mesh_space[1:-1] - mesh_space[:-2])

        # Compute diffusion matrix.
        diffusion = torch.zeros(3, mesh_space[1:-1].size(0))
        diffusion[0, :-1] = fwd[:-1]
        diffusion[1, :] = -fwd - bwd
        diffusion[2, 1:] = bwd[1:]
        return diffusion

    def stochastic_integral(self, time_0: torch.tensor, time_1: torch.tensor, mesh_space: torch.tensor,
                            density: torch.tensor, brownian_increment: torch.tensor,
                            method: str = 'milstein'):
        fwd = 1/(2*(mesh_space[2:] - mesh_space[1:-1]))
        bwd = 1/(2*(mesh_space[1:-1] - mesh_space[:-2]))
        density_0 = torch.zeros(density.size(0), density.size(1) + 2)
        density_0[:, 1:-1] = density
        diff = fwd*(density_0[:, 2:] - density_0[:, 1:-1]) + bwd*(density_0[:, 1:-1] - density_0[:, :-2])

        if method == 'euler':
            return (self.volatility_0*brownian_increment).unsqueeze(-1)*diff
        else:
            diff_2 = torch.zeros(density.size(0), density.size(1) + 2)
            diff_2[:, 1:-1] = diff
            delta_time = (self.volatility_0**2*(brownian_increment**2 - (time_1 - time_0))/2).unsqueeze(-1)
            diff_2 = delta_time*(fwd*(diff_2[:, 2:] - diff_2[:, 1:-1]) + bwd*(diff_2[:, 1:-1] - diff_2[:, :-2]))

            diff = (self.volatility_0*brownian_increment).unsqueeze(-1)*diff

            return diff - diff_2

    def forward(self, mesh_time: torch.tensor, mesh_space: torch.tensor, initial_condition: torch.tensor,
                common_noise: Optional[Union[int, torch.tensor]] = None, method: str = 'milstein',
                trajectories: bool = True
                ) -> Union[Tuple[torch.tensor, torch.tensor, torch.tensor], Tuple[torch.tensor, torch.tensor]]:
        """
        Numerically computes a discretisation of the solution to one-dimensional Fokker-Planck equation with controlled
        drift, feedback, and absorption as well as the associated cost. The discretisastion is on mesh_time and
        mesh_space and has initial condition initial_condition.

        :param mesh_time: mesh for time discretisation; (num_t, )
        :param mesh_space: mesh for space discretisation; (num_x, )
        :param initial_condition: initial condition of pde; (num_x - 2, ) or (num_samples, num_x - 2)
        :param common_noise: if torch.tensor then they represent increments of common Brownian motion, if int
        then it represents number of samples to be generated for common Brownian motion, if None then method
        assumes no common Brownian motion: (num_samples, num_t) or int or None
        :param method: SDE discretisation scheme; 'milstein' or 'euler'
        :param trajectories: if trajectories = True keep entire trajectories of density, otherwise only density at
        final time
        :return: density, cost: solution of pde and associated cost; (num_t, num_x - 2), (0)
        """
        # Sample increments of Brownian motion if no increments are given, otherwise extract number of samples from
        # Brownian increments.
        if common_noise is None:
            num_samples = 1
        elif type(common_noise) == int:
            num_samples = common_noise
            brownian_increment = (mesh_time[1:] - mesh_time[:-1])**(1/2)*torch.randn(num_samples, mesh_time.size(0) - 1)
        else:
            num_samples = common_noise.size(0)
            brownian_increment = common_noise

        # Initialise density and cost tensors.
        if trajectories:
            density = torch.zeros(num_samples, mesh_time.size(0), initial_condition.size(-1))
            density[:, 0, :] = initial_condition
        else:
            density = torch.zeros(num_samples, initial_condition.size(-1))
            density[:, :] = initial_condition

        cost = torch.zeros(mesh_time.size(0))
        # TODO: change back.
        # self.control.initialise()

        for i in range(0, mesh_time.size(0) - 1):
            time_0 = mesh_time[i]
            time_1 = mesh_time[i + 1]
            if trajectories:
                density_0 = density[:, i, :]
            else:
                density_0 = density

            # TODO: change back.
            # Compute control and running cost.
            control = self.control(time_0, mesh_space, density_0)
            # control = self.control(time_0, mesh_space, time_1 - time_0, brownian_increment[:, i])
            cost[i] = (time_1 - time_0)*ut.integrate(self.running_cost(control), density_0, mesh_space).mean(0)

            # Compute absorption, drift, and diffusion matrix, and compute system matrix
            absorption = self.absorption(time_0, time_1, mesh_space)
            drift = self.drift(time_0, time_1, mesh_space, density_0, control)
            diffusion = self.diffusion(time_0, time_1, mesh_space)
            eye = torch.zeros(3, density.size(-1))
            eye[1, :] = 1.
            diagonals = eye - absorption - drift - diffusion

            # Compute the increment stochastic integral and solve for next time step.
            if common_noise is not None:
                stochastic_integral = self.stochastic_integral(time_0, time_1, mesh_space, density_0,
                                                               brownian_increment[:, i], method=method)
                density_0 = solve(diagonals, density_0 - stochastic_integral)
                # density_0 = solve(diagonals, density_0)
            else:
                density_0 = solve(diagonals, density_0)

            if trajectories:
                density[:, i + 1, :] = density_0
            else:
                density = density_0

        # Compute terminal cost.
        if trajectories:
            cost[-1] = ut.integrate(self.terminal_cost(mesh_space[1:-1]).unsqueeze(0),
                                    density[:, -1, :], mesh_space).mean(0)
        else:
            cost[-1] = ut.integrate(self.terminal_cost(mesh_space[1:-1]).unsqueeze(0), density, mesh_space).mean(0)

        cost = torch.concat([cost[:-1].sum().unsqueeze(dim=0), cost[-1].unsqueeze(dim=0)], dim=0)
        if common_noise is not None:
            return density, cost, brownian_increment
        else:
            return density.squeeze(0), cost

    def fit(self, mesh_time: torch.tensor, mesh_space: torch.tensor, initial_condition: torch.tensor, num_samples: int,
            num_epochs: int, lr: float = 0.01, print_frequency: int = 1, plot: bool = False):
        """
        Trains the FinDiffScheme fin_diff_scheme for num_epochs with learning rate lr.

        :param mesh_time: mesh for time discretisation in fin_diff_scheme; (num_t, )
        :param mesh_space: mesh for space discretisation in fin_diff_scheme; (num_x, )
        :param initial_condition: initial condition of pde; (num_x - 2, ) or (num_samples, num_x - 2)
        :param num_samples: number of Brownian paths; int
        :param num_epochs: number of epochs; int
        :param lr: learning rate; float
        :param print_frequency: number of epochs between print statements containing cost; int
        :param plot: if True plot loss and heatmap of density every frequency epoch; otherwise no plots
        :return: costs: tensor with costs during given epoch; (num_epochs, )
        """

        costs = torch.zeros(num_epochs, requires_grad=False)
        optimiser = torch.optim.Adam(params=self.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimiser, factor=0.1, patience=10)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimiser, T_max=num_epochs, eta_min=lr/100)
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimiser, total_iters=num_epochs)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimiser, step_size=100, gamma=0.1)

        for i in range(num_epochs):
            # Solve SDE from 0 to final_time.
            density, cost, _ = self.forward(
                mesh_time, mesh_space, initial_condition, common_noise=num_samples, trajectories=plot)
            cost = cost[0] + cost[1]

            optimiser.zero_grad()
            cost.backward()
            optimiser.step()

            costs[i] = cost.detach()

            if (i + 1) % print_frequency == 0:
                current_lr = [group['lr'] for group in optimiser.param_groups][0]
                cost_avg = costs[i + 1 - print_frequency:i + 1].mean()
                logging.info(f"Epoch {i + 1}: Cost = {cost_avg:.5}, Learning Rate = {current_lr}")

                if plot:
                    loss = 1 - (density*(mesh_space[2:] - mesh_space[1:-1])).sum(-1).mean(0)
                    plt.plot(mesh_time, loss.detach())
                    plt.show()

                    fig, ax = ut.heatplot(density[0, :, :], mesh_time, mesh_space)
                    fig.show()

            # scheduler.step()

        return costs, [group['lr'] for group in optimiser.param_groups][0]

    def clone(self, mesh_time: torch.tensor, mesh_space: torch.tensor, initial_condition: torch.tensor,
              control: Callable, num_samples: int, num_iters: int, num_epochs: int, batch_size: int, lr: float = 0.001,
              print_frequency: int = 10, plot: bool = False):
        costs = torch.zeros(num_iters, num_epochs, requires_grad=False)
        optimiser = torch.optim.Adam(params=self.parameters(), lr=lr, weight_decay=10.)

        for i in range(num_iters):
            # TODO: change back.
            # self.control.initialise()

            # Solve SDE from 0 to final_time.
            densities, _, _ = self.forward(mesh_time, mesh_space, initial_condition, common_noise=num_samples,
                                           trajectories=True)
            times = torch.tile(mesh_time, (num_samples, 1)).reshape(num_samples*mesh_time.size(0))
            dataset = torch.utils.data.TensorDataset(
                densities.reshape(num_samples*mesh_time.size(0), mesh_space.size(0) - 2).detach(), times)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

            for j in range(num_epochs):
                # TODO: change back.
                # self.control.initialise()
                loss_mean = torch.tensor(0.)
                for density, time in dataloader:
                    # TODO: change back.
                    loss = ((control(mesh_space[1:-1])
                             - self.control(time, mesh_space, density.detach())).norm(p=2, dim=None))**2
                    # loss = ((control(mesh_space[1:-1])
                    #          - self.control.readout(time, mesh_space)).norm(p=2, dim=None))**2
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    loss_mean = loss_mean + loss
                costs[i, j] = loss_mean/(num_samples*mesh_time.size(0))

                if (j + 1) % print_frequency == 0:
                    logging.info(f'Iteration {i + 1}, Epoch {j + 1}: Loss = {costs[i, j]}')

                    if plot:
                        # Plot control and density.
                        values = torch.zeros(mesh_time.size(0), mesh_space.size(0) - 2)
                        for t in range(mesh_time.size(0)):
                            # TODO: change back.
                            values[t, :] = self.control(mesh_time[t], mesh_space, densities[:1, t, :]).detach()
                            # values[t, :] = self.control.readout(mesh_time[t], mesh_space)[0, :].detach()

                        mesh_x, mesh_t = np.meshgrid(mesh_time, mesh_space[1:-1])
                        fig, ax = plt.subplots()
                        ax.pcolormesh(mesh_x, mesh_t, values.transpose(0, 1), cmap='plasma')

                        c = ax.pcolormesh(mesh_x, mesh_t, values.transpose(0, 1), cmap='plasma')
                        fig.colorbar(c, ax=ax)
                        plt.show()

                        # mesh_x, mesh_t = np.meshgrid(mesh_time, mesh_space[1:-1])
                        # fig, ax = plt.subplots()
                        # ax.pcolormesh(mesh_x, mesh_t, densities[0, :, :].detach().transpose(0, 1), cmap='plasma')
                        #
                        # c = ax.pcolormesh(mesh_x, mesh_t, densities[0, :, :].detach().transpose(0, 1), cmap='plasma')
                        # fig.colorbar(c, ax=ax)
                        # plt.show()

    def multilevel_fit(self, meshs_time: List, meshs_space: List, initial_conditions: List, num_samples: torch.tensor,
                       num_epochs: torch.tensor, lr: float = 0.001, print_frequency: int = 10, plot: bool = False):
        costs = torch.tensor([], requires_grad=False)
        for i in range(len(meshs_time)):
            logging.info(f"Training for {num_epochs[i]} epochs with time steps = {meshs_time[i].size(0)}, "
                         f"space steps = {meshs_space[i].size(0)}, samples = {num_samples[i]}.")
            costs_i, lr = self.fit(meshs_time[i], meshs_space[i], initial_conditions[i], num_samples[i], num_epochs[i],
                                   lr=lr, print_frequency=print_frequency, plot=plot)
            costs = torch.cat((costs, costs_i))

            # TODO: decide whether to set multiply learning rate by 10 before going into next iteration of loop.

        return costs

    def get_state(self):
        state = {
            "volatility": self.volatility,
            "volatility_0": self.volatility_0,
            "feedback": self.feedback,
            "intensity_parameter": self.intensity_parameter,
            "intensity_type": self.intensity_type,
            "mean_reversion": self.mean_reversion,
            "control": self.control.state_dict()
        }
        return state

    @classmethod
    def load_state(cls, state, control, running_cost, terminal_cost):
        control.load_state_dict(state["control"])
        fin_diff_scheme = cls(control, running_cost, terminal_cost)
        fin_diff_scheme.initialise_parameters(state["volatility"], state["volatility_0"], state["feedback"],
                                              state["intensity_parameter"], state["intensity_type"])
        return fin_diff_scheme

