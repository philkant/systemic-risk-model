import torch
from matplotlib import pyplot as plt
from typing import Callable, Tuple

from systemic_risk_model import controls as ct, finite_element as fd

torch.set_default_dtype(torch.float64)
path = '/Users/philipp/PycharmProjects/systemic_risk_model/data/'


def constant_control(volatility: float, volatility_0: float, feedback: float, intensity: float, mesh_time: torch.tensor,
                     mesh_space: torch.tensor, running_cost: Callable, terminal_cost: Callable,
                     initial_condition: torch.tensor, common_noise: bool = True, lower: float = 0.5, upper: float = 0.5,
                     num: int = 20):
    torch.manual_seed(0)
    constants = torch.linspace(lower, upper, num)
    costs = torch.zeros(num, requires_grad=False)
    for i in range(0, num):
        print(i)
        control = ct.ConstantControl(constants[i].item())

        fin_diff_scheme = fd.FinDiffScheme(control, running_cost, terminal_cost)
        fin_diff_scheme.initialise_parameters(volatility, volatility_0, feedback, intensity)
        density, cost, brownian_motion = fin_diff_scheme(mesh_time, mesh_space, initial_condition,
                                                         common_noise=int(common_noise))
        costs[i] = cost.detach()

    plt.plot(constants.detach(), costs.detach())
    plt.show()


def fit(volatility: float, volatility_0: float, feedback: float, intensity: float, mesh_time: torch.tensor,
        mesh_space: torch.tensor, running_cost: Callable, terminal_cost: Callable,  control: ct.Control,
        initial_condition: torch.tensor, num_samples: int, num_epochs: int):

    fin_diff_scheme = fd.FinDiffScheme(control, running_cost, terminal_cost)
    # fin_diff_scheme.load_state_dict(torch.load('/Users/philipp/PycharmProjects/systemic_risk_model/fin_diff_scheme.pt'))
    fin_diff_scheme.initialise_parameters(volatility, volatility_0, feedback, intensity)
    costs = fin_diff_scheme.fit(mesh_time, mesh_space, initial_condition, num_samples, num_epochs)
    plt.plot(range(0, costs.size(0)), costs)
    plt.show()

    torch.save(fin_diff_scheme.state_dict(), path + 'fin_diff_scheme.pt')


def capital_requirements(volatility: float, volatility_0: float, cap_reqs: torch.tensor, intensity: float,
                         discount_0: float, discount_1: float, innerbank_liabilities: float, mesh_time: torch.tensor,
                         mesh_space: torch.tensor, running_cost: Callable, terminal_cost: Callable, control: ct.Control,
                         initial_condition: torch.tensor, num_samples: int, num_epochs: int
                         ) -> Tuple[torch.tensor, torch.tensor]:
    costs_gd = torch.zeros(cap_reqs.size(0), num_epochs)
    densities = torch.zeros(cap_reqs.size(0), num_samples, mesh_time.size(0), mesh_space.size(0) - 2)
    costs = torch.zeros(cap_reqs.size(0), 2)

    for i in range(cap_reqs.size(0)):
        # Rest seed and weights of control.
        torch.manual_seed(0)
        control.apply(ct.weight_reset)

        # Initialise parameters for finite difference scheme.
        def feedback(loss: torch.tensor) -> torch.tensor:
            return torch.maximum((discount_0 - cap_reqs[i])/(1 - cap_reqs[i]) + discount_1/(1 - cap_reqs[i])*loss,
                                 torch.tensor(0.))*innerbank_liabilities

        fin_diff_scheme = fd.FinDiffScheme(control, running_cost, terminal_cost)
        fin_diff_scheme.initialise_parameters(volatility, volatility_0, feedback, intensity)

        # Train control and return cost for trained control.
        costs_gd[i, :] = fin_diff_scheme.fit(mesh_time, mesh_space, initial_condition, num_samples, num_epochs,
                                             print_frequency=1).detach()
        # plt.plot(range(num_epochs), costs_gd[i, :])
        # plt.show()

        density, cost, _ = fin_diff_scheme(mesh_time, mesh_space, initial_condition, common_noise=num_samples)
        densities[i, ...] = density.detach()
        costs[i, :] = cost.detach()

        torch.save(fin_diff_scheme.state_dict(), path + 'feedback_model_{}.pt'.format(i))

    torch.save(costs_gd, path + 'feedback_costs_gd.pt')
    torch.save(densities, path + 'feedback_densities.pt')
    torch.save(costs, path + 'feedback_costs.pt')
    plt.plot(cap_reqs, costs[:, 0], label='running cost')
    plt.plot(cap_reqs, 1 + costs[:, 0] + costs[:, 1], label='total cost')
    plt.legend()
    plt.show()

    return densities, costs


def volatility_parameter(volatility: float, volatilities_0: torch.tensor, feedback: float, intensity: float,
                         mesh_time: torch.tensor, mesh_space: torch.tensor, running_cost: Callable,
                         terminal_cost: Callable, control: ct.Control, initial_condition: torch.tensor,
                         num_samples: int, num_epochs: int):

    costs_gd = torch.zeros(volatilities_0.size(0), num_epochs)
    densities = torch.zeros(volatilities_0.size(0), num_samples, mesh_time.size(0), mesh_space.size(0) - 2)
    costs = torch.zeros(volatilities_0.size(0))

    for i in range(volatilities_0.size(0)):
        # Rest seed and weights of control.
        torch.manual_seed(0)
        control.apply(ct.weight_reset)

        # Initialise parameters for finite difference scheme.
        fin_diff_scheme = fd.FinDiffScheme(control, running_cost, terminal_cost)
        fin_diff_scheme.initialise_parameters(volatility, volatilities_0[i], feedback, intensity)

        # Train control and return cost for trained control.
        costs_gd[i, :] = fin_diff_scheme.fit(mesh_time, mesh_space, initial_condition, num_samples, num_epochs).detach()
        plt.plot(range(num_epochs), costs_gd[i, :])
        plt.show()

        density, cost, _ = fin_diff_scheme(mesh_time, mesh_space, initial_condition, common_noise=num_samples)
        densities[i, ...] = density.detach()
        costs[i] = cost.detach()

        torch.save(fin_diff_scheme.state_dict(), path + 'volatility_model_{}.pt'.format(i))

    torch.save(costs_gd, path + 'volatility_costs_gd.pt')
    torch.save(densities, path + 'volatility_densities.pt')
    torch.save(costs, path + 'volatility_costs.pt')

    return densities, costs


def capital_requirement(volatility: float, volatility_0: float, intensity: float, mesh_time: torch.tensor,
                        mesh_space: torch.tensor, running_cost: Callable, terminal_cost: Callable, control: ct.Control,
                        excess_equity: torch.tensor, interbank_liabilities: float, cap_reqs: torch.tensor,
                        cap_req_penalty: float, num_samples: int, num_epochs: int):
    # Set capital requirements and feedback parameter.
    lower = cap_reqs[0]
    upper = cap_reqs[-1]

    cap_req = torch.linspace(lower, upper, num_samples)
    feedback = interbank_liabilities*(1 - cap_req)**2

    # Initialise parameters for finite difference scheme and capital requirements for control.
    control.initialise_cap_req(cap_req)
    fin_diff_scheme = fd.FinDiffScheme(control, running_cost, terminal_cost)
    fin_diff_scheme.initialise_parameters(volatility, volatility_0, feedback, intensity)

    # Train control and return cost of gradient iterations.
    costs = fin_diff_scheme.fit(mesh_time, mesh_space, excess_equity, num_samples, num_epochs)
    plt.plot(range(0, costs.size(0)), costs)
    plt.show()

    # control = ct.ConstantControl(0.)
    # fin_diff_scheme = fd.FinDiffScheme(control, running_cost, terminal_cost)

    densities = torch.zeros(cap_reqs.size(0), num_samples, mesh_time.size(0), mesh_space.size(0) - 2)
    costs = torch.zeros(cap_reqs.size(0))
    brownian_increments = (mesh_time[1:] - mesh_time[:-1])**(1/2)*torch.randn(num_samples, mesh_time.size(0) - 1)
    for i in range(costs.size(0)):
        # Set capital requirements and feedback parameter.
        cap_req = torch.full((num_samples, ), cap_reqs[i])
        feedback = interbank_liabilities*(1 - cap_req)**2

        # Initialise parameters for finite difference scheme and capital requirements for control.
        fin_diff_scheme.control.initialise_cap_req(cap_req)
        fin_diff_scheme.initialise_parameters(volatility, volatility_0, feedback, intensity)
        density, cost, _ = fin_diff_scheme.forward(mesh_time, mesh_space, excess_equity,
                                                   common_noise=brownian_increments)
        densities[i, ...] = density.detach()
        costs[i] = cost.detach()

        # fig, ax = ut.heatplot(density[0, ...], mesh_time, mesh_space)
        # fig.show()
        torch.save(fin_diff_scheme.state_dict(), path + 'cap_req_model_{}.pt'.format(i))

    torch.save(densities, path + 'cap_req_densities.pt')
    torch.save(costs, path + 'cap_req_costs.pt')

    plt.plot(cap_reqs, costs.detach() + cap_req_penalty*cap_reqs)
    plt.show()
