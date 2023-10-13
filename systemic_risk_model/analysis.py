import os
import torch
import fnmatch
import numpy as np

from matplotlib import pyplot as plt
from typing import List, Optional, Union

from systemic_risk_model import finite_element as fd, controls as ct

torch.set_default_dtype(torch.float64)


def num_outputs(experiment_name: str):
    num = 0
    for file in os.listdir('./data'):
        if fnmatch.fnmatch(file, 'output_' + experiment_name + '*'):
            num += 1

    return num


def get_data(experiment: str, fin_diff_scheme: fd.FinDiffScheme, mesh_time: torch.tensor, mesh_space: torch.tensor,
             num_samples: int, num_epochs: List):
    num_parameters = num_outputs(experiment)
    parameters = torch.zeros(num_parameters)
    costs_training = torch.zeros(num_parameters, sum(num_epochs))
    costs = torch.zeros(num_parameters, 2)
    densities = torch.tensor([])
    losses = torch.tensor([])
    controls = torch.zeros(num_parameters, num_samples, mesh_time.size(0), mesh_space.size(0) - 2)

    if experiment == 'intensity':
        parameter = 'intensity_parameter'
    elif experiment == 'volatility':
        parameter = 'volatility_0'
    else:
        parameter = experiment

    for i in range(num_parameters):
        # Get feedback parameters, costs, densities, loss, and mean of value of controls.
        output = torch.load(f'./data/output_{experiment}_{i}.pt')
        if experiment == 'correlation':
            volatility = output['fin_diff_scheme_state']['volatility']
            volatility_0 = output['fin_diff_scheme_state']['volatility_0']
            parameters[i] = round(volatility_0**2/(volatility**2 + volatility_0**2), 2)
        else:
            parameters[i] = output['fin_diff_scheme_state'][parameter]
        costs_training[i, :] = output['cost_training']
        costs[i, :] = output['cost']
        densities = torch.cat((densities, output['density'].unsqueeze(0)), dim=0)
        loss = 1 - (output['density']*(mesh_space[2:] - mesh_space[1:-1])).sum(-1)
        losses = torch.cat((losses, loss.unsqueeze(0)), dim=0)

        # TODO: change back
        # fin_diff_scheme.control.initialise()
        fin_diff_scheme.control.load_state_dict(output["fin_diff_scheme_state"]["control"])
        for t in range(mesh_time.size(0)):
            controls[i, :, t, :] = fin_diff_scheme.control(
                mesh_time[t], mesh_space, densities[i, :, t, :]).detach()

    densities = densities.relu()
    controls = controls.relu()

    return parameters, costs_training, costs, densities, losses, controls


def section(values: torch.tensor, mesh_space:torch.tensor, x_min: float, x_max: float):
    idx_min = torch.argmin(torch.abs(mesh_space[1:-1] - x_min)).item()
    idx_max = torch.argmin(torch.abs(mesh_space[1:-1] - x_max)).item()

    values_section = values[..., idx_min:idx_max + 1]
    mesh_space_section = mesh_space[idx_min:idx_max + 3]

    return values_section, mesh_space_section


def heatplot(values: torch.tensor, mesh_time: torch.tensor, mesh_space: torch.tensor,
             value_min: Optional[float] = None, value_max: Optional[float] = None):

    if value_min is None:
        value_min = values.min()
    if value_max is None:
        value_max = values.max()

    mesh_x, mesh_t = np.meshgrid(mesh_time, mesh_space[1:-1])
    fig, ax = plt.subplots()
    ax.pcolormesh(mesh_x, mesh_t, values.transpose(0, 1), cmap='plasma')

    c = ax.pcolormesh(mesh_x, mesh_t, values.transpose(0, 1),
                      cmap='plasma', vmin=value_min, vmax=value_max)
    fig.colorbar(c, ax=ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('Distance-to-Breach')

    return fig, ax


def heatplots(parameters: torch.tensor, symbol: str, y_label: str, values: torch.tensor, mesh_time: torch.tensor,
              mesh_space: torch.tensor, value_min: Optional[float] = None, value_max: Optional[float] = None,
              share=True):

    if value_min is None:
        value_min = values.min()
    if value_max is None:
        value_max = values.max()

    cols = values.size(0)
    rows = values.size(1)

    mesh_x, mesh_t = np.meshgrid(mesh_time, mesh_space[1:-1])
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)

    c = None

    if rows == 1:
        for col_idx, ax in enumerate(axes):
            ax.set_aspect('equal')
            c = ax.pcolormesh(mesh_x, mesh_t, values[col_idx, 0, ...].transpose(0, 1), cmap='plasma', vmin=value_min,
                              vmax=value_max)
            ax.set_title(fr"${symbol} = {parameters[col_idx]}$")
            ax.set_xlabel('Time')

            if col_idx == 0:
                ax.set_ylabel(y_label)

        fig.colorbar(c, ax=axes.ravel().tolist(), fraction=0.01034, pad=0.03)

    else:
        for row_idx, row in enumerate(axes):
            for col_idx, ax in enumerate(row):
                c = ax.pcolormesh(mesh_x, mesh_t, values[col_idx, row_idx, ...].transpose(0, 1), cmap='plasma',
                                  vmin=value_min, vmax=value_max)
                if not share and row_idx != 0:
                    ax.set_title(fr"${symbol} = {parameters[col_idx + cols*row_idx]}$")
                if row_idx == 0:
                    ax.set_title(fr"${symbol} = {parameters[col_idx]}$")
                if row_idx == rows - 1:
                    ax.set_xlabel('Time')
                if col_idx == 0:
                    ax.set_ylabel(y_label)

        fig.colorbar(c, ax=axes.ravel().tolist())
        plt.subplots_adjust(right=0.76, hspace=0.13, wspace=0.13)

    return fig, axes


def plot_cost(parameters: Union[torch.tensor, List], parameter_name: str, parameter_symbol: str, costs: torch.tensor,
              ax=None):
    if type(parameters) == list:
        parameters_list = parameters
    else:
        parameters_list = parameters.tolist()

    if ax is None:
        _, ax = plt.subplots()

    width = .35
    ax.bar(range(len(parameters_list)), costs[:, 0], width, label='Running cost')
    ax.bar(range(len(parameters_list)), costs[:, 1], width, bottom=costs[:, 0], label='Terminal cost')

    ticks = []
    for i in range(len(parameters_list)):
        ticks += [r'${} = {}$'.format(parameter_symbol, parameters_list[i])]
    ax.set_xticks(range(len(parameters_list)), ticks)
    ax.legend()
    ax.set_ylabel('Cost')
    ax.set_title(f'Cost for different {parameter_name}')

    return ax


def plot_total_cost(parameters: Union[torch.tensor, List], parameter_name: str, parameter_symbol: str,
                    costs: torch.tensor):
    if type(parameters) == list:
        parameters_list = parameters
    else:
        parameters_list = parameters.tolist()

    fig, ax = plt.subplots()

    ax.plot(range(len(parameters_list)), costs)

    ticks = []
    for i in range(len(parameters_list)):
        ticks += [r'${} = {}$'.format(parameter_symbol, parameters_list[i])]
    ax.set_xticks(range(len(parameters_list)), ticks)
    ax.legend()
    ax.set_ylabel('Cost')
    ax.set_title(f'Total cost for different {parameter_name}')

    return fig, ax


def plot_cost_training(costs: torch.tensor, num_epochs: List, discretisations: List):
    fig, ax = plt.subplots()
    ax.plot(range(sum(num_epochs)), costs)
    loc = 0
    for i in range(len(num_epochs)):
        ax.axvline(x=loc, linestyle='dashed', color='black', linewidth=0.5)
        # ax.text(loc, costs.max() - 0.1, ' ' + str(discretisations[i]), va='bottom', ha='left', color='black')
        loc += num_epochs[i]
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost')
    ax.set_title('Convergence of SGD')

    return fig, ax


def plot_loss(parameters: Union[torch.tensor, List], parameter_name: str, parameter_symbol: str,
              mesh_time: torch.tensor, losses: torch.tensor):
    if type(parameters) == list:
        parameters_list = parameters
    else:
        parameters_list = parameters.tolist()

    fig, ax = plt.subplots()
    for i in range(len(parameters_list)):
        ax.plot(mesh_time, losses[i, :], label=r'${} = {}$'.format(parameter_symbol, parameters_list[i]))
        # ax.plot(mesh_time, losses[i, 0, :], label=r'${} = {}$'.format(parameter_symbol, parameters_list[i]))
        # ax.plot(mesh_time, losses[i, 1, :], label=r'${} = {}$'.format(parameter_symbol, parameters_list[i]))
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss for different {parameter_name}')

    return fig, ax


def plot_intensity_function(intensity_parameters: torch.tensor, intensity_types: List, mesh_space: torch.tensor):
    fig, ax = plt.subplots()
    for i in range(intensity_parameters.size(0)):
        intensity_function = fd.get_intensity(intensity_parameters[i], intensity_types[i])
        ax.plot(mesh_space[1:-1], intensity_function(mesh_space[1:-1]), label=intensity_types[i])

    ax.set_xlabel('Space')
    ax.set_ylabel('Intensity')
    ax.set_title('Plot of intensity functions')
    ax.legend()

    return fig, ax


def get_symbol(experiment: str):
    if experiment == 'feedback':
        return r'\alpha'
    elif experiment == 'volatility':
        return r'\sigma_0'
    elif experiment == 'intensity':
        return r'\lambda'
    elif experiment == 'mean_reversion':
        return r'\mu'
    elif experiment == 'correlation':
        return r'\rho'


def get_plural(experiment: str):
    if experiment == 'feedback':
        return 'feedbacks'
    elif experiment == 'volatility':
        return 'volatilities'
    elif experiment == 'intensity':
        return 'intensities'
    elif experiment == 'mean_reversion':
        return 'mean reversions'
    elif experiment == 'correlation':
        return 'correlations'


def analysis(experiment: str, fin_diff_scheme: fd.FinDiffScheme, mesh_time: torch.tensor,
             mesh_space: torch.tensor, num_samples: int, num_epochs: List, discretisations: List):
    parameters, costs_training, costs, densities, losses, controls = get_data(
        experiment, fin_diff_scheme, mesh_time, mesh_space, num_samples, num_epochs)
    densities = torch.maximum(densities, torch.tensor(0.))
    symbol = get_symbol(experiment)

    if experiment == 'feedback':
        para_idx = [0, parameters.size(0) - 1]
        path_idx = [13, 15]
        analysis_standard(experiment, parameters, symbol, costs_training, costs, densities, losses, controls, mesh_time,
                          mesh_space, num_epochs, para_idx, path_idx, discretisations)

    elif experiment == 'volatility':
        para_idx = [1, parameters.size(0) - 1]
        path_idx = [13, 15]
        analysis_standard(experiment, parameters, symbol, costs_training, costs, densities, losses, controls, mesh_time,
                          mesh_space, num_epochs, para_idx, path_idx, discretisations)

    elif experiment == 'correlation':
        para_idx = [0, parameters.size(0) - 1]
        path_idx = [5, 15]
        analysis_standard(experiment, parameters, symbol, costs_training, costs, densities, losses, controls, mesh_time,
                          mesh_space, num_epochs, para_idx, path_idx, discretisations)

    elif experiment == 'intensity':
        para_idx = range(parameters.size(0))
        densities = densities[:, 11, :, :].unsqueeze(1)
        controls = controls[:, 11, :, :].unsqueeze(1)

        analysis_intensity(parameters, symbol, costs, controls, densities, mesh_time, mesh_space, para_idx)

    elif experiment == 'mean_reversion':
        para_idx = [0, parameters.size(0) - 1]
        path_idx = [11, 12]
        # high: 3
        # high: 5
        # high: 9
        # low: 11
        # low: 12
        # low: 15
        # high: 13
        # analysis_mean_reversion(parameters, symbol, costs, densities, losses, controls, mesh_time, mesh_space, para_idx)
        analysis_standard(experiment, parameters, symbol, costs_training, costs, densities, losses, controls, mesh_time,
                          mesh_space, num_epochs, para_idx, path_idx, discretisations)
    else:
        raise RuntimeError('Experiment not permitted.')

    return


def analysis_standard(experiment: str, parameters: torch.tensor, symbol: str, costs_training: torch.tensor,
                      costs: torch.tensor, densities: torch.tensor, losses: torch.tensor, controls: torch.tensor,
                      mesh_time: torch.tensor, mesh_space: torch.tensor, num_epochs: List[int], para_idx: List[int],
                      path_idx: List[int], discretisations: List[int], plot: bool = True):
    # TODO: remove if other terminal cost
    costs[:, 1] = costs[:, 1] + 1
    plot_cost(parameters, get_plural(experiment), symbol, costs)

    if plot:
        plt.show()
    else:
        plt.savefig('./data/plot_' + experiment + '_cost')

    # plot_loss(parameters, get_plural(experiment), symbol, mesh_time, losses.mean(1))
    # plt.show()

    costs_training = costs_training + 1
    plot_cost_training(costs_training[-1, :], num_epochs, discretisations)

    if plot:
        plt.show()
    else:
        plt.savefig('./data/plot_' + experiment + '_cost_training')

    x_min = -0.5
    x_max = 0.5
    densities_0, mesh_space_section = section(densities[para_idx, :, :, :][:, path_idx, :, :], mesh_space, x_min, x_max)

    fig, _ = heatplots(parameters[para_idx], symbol, 'Distance-to-breach', densities_0, mesh_time, mesh_space_section,
                       densities_0.min(), densities_0.max())
    fig.suptitle(r'Heat plot of subdistribution $\nu$')

    if plot:
        plt.show()
    else:
        plt.savefig('./data/plot_' + experiment + '_density')

    controls_0, mesh_space_section = section(controls[para_idx, :, :, :][:, path_idx, :, :], mesh_space, x_min, x_max)
    fig, _ = heatplots(parameters[para_idx], symbol, 'Distance-to-breach', controls_0, mesh_time, mesh_space_section,
                       controls_0.min(), controls_0.max())
    fig.suptitle(r'Heat plot of control $\gamma$')

    if plot:
        plt.show()
    else:
        plt.savefig('./data/plot_' + experiment + '_control')


# def analysis_feedback(parameters: torch.tensor, symbol: str, costs_training: torch.tensor, costs: torch.tensor,
#                       densities: torch.tensor, losses: torch.tensor, controls: torch.tensor, mesh_time: torch.tensor,
#                       mesh_space: torch.tensor, num_epochs: List[int], para_idx: List[int], path_idx: List[int]):
#     _, axes = plt.subplots(1, 2)
#
#     # TODO: remove if other terminal cost
#     costs[:, 1] = costs[:, 1] + 1
#     plot_cost(parameters, get_plural('feedback'), symbol, costs, ax=axes[1])
#
#     costs_training = costs_training + 1
#     plot_cost_training(costs_training[-1, :], num_epochs, ax=axes[0])
#
#     plt.show()
#
#     x_min = -0.5
#     x_max = 0.5
#     densities_0, mesh_space_section = section(densities[para_idx, :, :, :][:, path_idx, :, :], mesh_space, x_min, x_max)
#
#     fig, _ = heatplots(parameters[para_idx], symbol, 'Distance-to-breach', densities_0, mesh_time, mesh_space_section,
#                        densities_0.min(), densities_0.max())
#     fig.suptitle('Heat plot of densities')
#     plt.show()
#
#     controls_0, mesh_space_section = section(controls[para_idx, :, :, :][:, path_idx, :, :], mesh_space, x_min, x_max)
#     fig, _ = heatplots(parameters[para_idx], symbol, 'Control', controls_0, mesh_time, mesh_space_section,
#                        controls_0.min(), controls_0.max())
#     fig.suptitle('Heat plot of controls')
#     plt.show()


def analysis_intensity(parameters: torch.tensor, symbol: str, costs: torch.tensor, controls: torch.tensor,
                       densities: torch.tensor, mesh_time: torch.tensor, mesh_space: torch.tensor,
                       para_idx: Union[List[int], range], plot=False):
    costs[:, 1] = costs[:, 1] + 1
    plot_cost(parameters, get_plural('intensity'), symbol, costs)

    if plot:
        plt.show()
    else:
        plt.savefig('./data/plot_intensity_cost')

    x_min = -0.5
    x_max = 0.5

    densities = densities[para_idx, :, :, :].squeeze(1).reshape(2, 2, densities.size(-2), densities.size(-1))
    densities = densities.transpose(0, 1)
    densities_0, mesh_space_section = section(densities, mesh_space, x_min, x_max)
    fig, axes = heatplots(parameters[para_idx], symbol, 'Distance-to-breach', densities_0, mesh_time,
                          mesh_space_section, densities_0.min(), densities_0.max(), share=False)
    fig.suptitle(r'Heat plot of subdistribution $\nu$')

    plt.subplots_adjust(hspace=0.175)

    if plot:
        plt.show()
    else:
        plt.savefig('./data/plot_intensity_density')

    controls = controls[para_idx, :, :, :].squeeze(1).reshape(2, 2, controls.size(-2), controls.size(-1))
    controls = controls.transpose(0, 1)
    # controls_0, mesh_space_section = section(controls[para_idx, :, :, :], mesh_space, x_min, x_max)
    # fig, axes = heatplots(parameters[para_idx], symbol, 'Distance-to-breach', controls_0, mesh_time, mesh_space_section,
    #                       controls_0.min(), controls_0.max())
    # fig.suptitle(r'Heat plot of control $\gamma$', y=0.75)
    controls_0, mesh_space_section = section(controls, mesh_space, x_min, x_max)
    fig, axes = heatplots(parameters[para_idx], symbol, 'Distance-to-breach', controls_0, mesh_time, mesh_space_section,
                          controls_0.min(), controls_0.max(), share=False)
    fig.suptitle(r'Heat plot of control $\gamma$')

    # if plot:
    #     plt.show()
    # else:
    #     plt.savefig('./data/plot_intensity_control')

    # controls_0 = controls_0.squeeze(1)
    # # idx = torch.argmin(torch.abs(controls_0 - 0.05), dim=-1)
    # idx = torch.argmin(torch.relu(controls_0 - 0.05), dim=-1)
    # contour = mesh_space_section[1:-1][idx]

    controls_0 = controls_0.reshape(-1, controls_0.size(-2), controls_0.size(-1))
    # idx = torch.argmin(torch.abs(controls_0 - 0.05), dim=-1)
    idx = torch.argmin(torch.relu(controls_0 - 0.05), dim=-1)
    contour = mesh_space_section[1:-1][idx]

    for row_idx, row in enumerate(axes):
        for col_idx, ax in enumerate(row):
            ax.plot(mesh_time, contour[row_idx + 2*col_idx, :], c='C2')

    # for idx, ax in enumerate(axes):
    #     ax.plot(mesh_time, contour[idx, :], c='C2')
    plt.subplots_adjust(hspace=0.175)

    if plot:
        plt.show()
    else:
        plt.savefig('./data/plot_intensity_control')

    fig, ax = plt.subplots()
    for idx in range(len(para_idx)):
        ax.plot(mesh_time, contour[idx, :], c='C2')
    ax.set_xlabel('Time')
    ax.set_ylabel('Distance-to-breach')
    ax.set_title(r'Contour of control for intensities $\lambda = 5$, $10$, $25$, and $50$')

    if plot:
        plt.show()
    else:
        plt.savefig('./data/plot_intensity_control_contour')


def analysis_mean_reversion(parameters: torch.tensor, symbol: str, costs: torch.tensor, densities: torch.tensor,
                            losses: torch.tensor, controls: torch.tensor, mesh_time: torch.tensor,
                            mesh_space: torch.tensor, para_idx: Union[List[int], range], plot=False):
    costs[:, 1] = costs[:, 1] + 1
    plot_cost(parameters, get_plural('mean_reversion'), symbol, costs)
    plt.show()
    #
    means = (densities*(mesh_space[2:] - mesh_space[1:-1])*mesh_space[1:-1]).sum(-1)
    #
    plot_loss(parameters, 'Mean reversion', symbol, mesh_time, means[:, 2, :])
    plt.show()

    losses_cond_exp = torch.zeros(parameters.size(0))
    beta = 0.0

    width = .35

    for i in range(parameters.size(0)):
        idx = losses[i, :, -1] > beta
        losses_cond_exp[i] = losses[i, idx, -1].sum()/idx.sum(0)

    plt.bar(range(parameters.size(0)), losses_cond_exp, width)
    plt.show()

    # idx = losses[3, :, -1] > 0.2
    # for i in range(parameters.size(0)):
    #     plt.scatter(means[i, idx, -1]/(1 - losses[i, idx, -1]), losses[i, idx, -1])
    # plt.show()

    # densities_0 = densities[:, 0, :, :]
    # for i in range(num_parameters):
    #     heatplot(densities_0[i, ...], mesh_time, mesh_space, densities_0.min(), densities_0.max())
    #     plt.title(r'Heat plot of density with ${} =$'.format(symbol) + f'{parameters[i]}')
    #     plt.show()
    #
    # densities_0 = densities[:, k, :, :]
    # for i in range(num_parameters):
    #     heatplot(densities_0[i, ...], mesh_time, mesh_space, densities_0.min(), densities_0.max())
    #     plt.title(r'Heat plot of density with ${} =$'.format(symbol) + f'{parameters[i]}')
    #     plt.show()
    #
    # x_min = -0.5
    # x_max = 0.5
    # controls_0, mesh_space_section = section(controls[:, 0, :, :], mesh_space, x_min, x_max)
    # for i in range(num_parameters):
    #     heatplot(controls_0[i, ...], mesh_time, mesh_space_section, controls.min(), controls.max())
    #     plt.title(r'Heat plot of control with ${} =$'.format(symbol) + f'{parameters[i]}')
    #     plt.show()
    #
    # controls_0, mesh_space_section = section(controls[:, k, :, :], mesh_space, x_min, x_max)
    # for i in range(num_parameters):
    #     heatplot(controls_0[i, ...], mesh_time, mesh_space_section, controls.min(), controls.max())
    #     plt.title(r'Heat plot of control with ${} =$'.format(symbol) + f'{parameters[i]}')
    #     plt.show()


# def analysis(experiment_name: str, innerbank_liabilities: float, discount_rate: float,
#              control_type: str, hidden_size: int, hidden_size_scalar: int, mesh_time: torch.tensor,
#              mesh_space: torch.tensor, num_samples: int, num_epochs: List, fin_diff_scheme: fd.FinDiffScheme):
#     if experiment_name == 'feedback':
#         analysis_feedback(fin_diff_scheme, mesh_time, mesh_space, num_samples, num_epochs)
#     elif experiment_name == 'volatility':
#         analysis_volatility(fin_diff_scheme, mesh_time, mesh_space, num_epochs)
#     elif experiment_name == 'capital_requirement':
#         analysis_capital_requirement(fin_diff_scheme, innerbank_liabilities, discount_rate, mesh_time, mesh_space,
#                                      num_samples)
#     elif experiment_name == 'mean_reversion':
#         analysis_mean_reversion(fin_diff_scheme, control_type, hidden_size, hidden_size_scalar, mesh_time, mesh_space,
#                                 num_samples)
#     elif experiment_name == 'control':
#         analysis_control(fin_diff_scheme, control_type, hidden_size, hidden_size_scalar, mesh_time, mesh_space,
#                          num_samples)
#     elif experiment_name == 'intensity':
#         analysis_intensity(fin_diff_scheme, mesh_time, mesh_space, num_samples)

def analysis_capital_requirement(fin_diff_scheme: fd.FinDiffScheme, innerbank_liabilities: float, discount_rate: float,
                                 mesh_time: torch.tensor, mesh_space: torch.tensor):
    num_parameters = num_outputs('capital_requirement')
    capital_requirements = torch.zeros(num_parameters)
    costs = torch.zeros(num_parameters, 2)
    losses = torch.tensor([])

    for i in range(num_parameters):
        output = torch.load(f'./data/output_capital_requirement_{i}.pt')
        feedback = output['fin_diff_scheme_state']['feedback']
        capital_requirements[i] = (discount_rate*innerbank_liabilities - feedback)/(innerbank_liabilities - feedback)
        costs[i, :] = output['cost']
        loss = 1 - (output['density']*(mesh_space[2:] - mesh_space[1:-1])).sum(-1)
        losses = torch.cat((losses, loss.unsqueeze(0)), dim=0)

    # Plot cost and loss.
    penalty = 0.5
    fig, ax = plot_total_cost(capital_requirements, 'capital requirements', r'c',
                              costs.sum(-1) + penalty*capital_requirements)

    fig, ax = plot_loss(capital_requirements, 'capital requirements', r'c', mesh_time, losses)
    plt.show()


def analysis_control(fin_diff_scheme: fd.FinDiffScheme, control_type: str, hidden_size: int, hidden_size_scalar: int,
                     mesh_time: torch.tensor, mesh_space: torch.tensor):
    num_parameters = num_outputs('control')
    print(num_parameters)
    mean_reversion_speeds = torch.zeros(num_parameters)
    costs = torch.zeros(num_parameters, 2)
    densities = torch.tensor([])
    losses = torch.tensor([])
    controls = torch.zeros(num_parameters, mesh_time.size(0), mesh_space.size(0) - 2)
    controls_1 = torch.zeros(num_parameters, mesh_time.size(0), mesh_space.size(0) - 2)

    k = 1
    # num_parameters = num_outputs('control')
    # costs = torch.zeros(num_parameters, 2)
    # densities = torch.tensor([])
    # losses = torch.tensor([])
    # controls = torch.zeros(num_parameters, mesh_time.size(0), mesh_space.size(0) - 2)

    for i in range(num_parameters):
        output = torch.load(f'./data/output_control_{i}.pt')
        control = ct.get_control(control_type, hidden_size, hidden_size_scalar=hidden_size_scalar)
        densities = torch.cat((densities, output['density'].unsqueeze(0)), dim=0)
        fin_diff_scheme = fd.FinDiffScheme.load_state(output["fin_diff_scheme_state"], control, None, None)
        costs[i, :] = output['cost']
        loss = 1 - (output['density']*(mesh_space[2:] - mesh_space[1:-1])).sum(-1)
        losses = torch.cat((losses, loss.unsqueeze(0)), dim=0)

        fin_diff_scheme.control.load_state_dict(output["fin_diff_scheme_state"]["control"])
        for t in range(mesh_time.size(0)):
            controls[i, t, :] = fin_diff_scheme.control(
                mesh_time[t], mesh_space, densities[i, :1, t, :]).squeeze(0).detach()
            controls_1[i, t, :] = fin_diff_scheme.control(
                mesh_time[t], mesh_space, densities[i, k:k + 1, t, :]).squeeze(0).detach()

    # Plot cost and loss.
    # TODO: remove if other terminal cost
    # costs[:, 1] = costs[:, 1] + 1
    # fig, ax = plot_cost(['NeuralLearnedScalar', 'NeuralMeanReverting'], 'controls', r'type', costs)
    # plt.show()
    #
    # fig, ax = plot_loss(['NeuralLearnedScalar', 'NeuralMeanReverting'], 'controls', r'type', mesh_time, losses)
    # plt.show()

    costs[:, 1] = costs[:, 1] + 1
    fig, ax = plot_cost(mean_reversion_speeds, 'mean_reversion_speeds', r'\mu', costs)
    plt.show()

    fig, ax = plot_loss(mean_reversion_speeds, 'mean_reversion_speeds', r'\mu', mesh_time, losses)
    plt.show()

    densities = torch.maximum(densities, torch.tensor(0.))
    densities_0 = densities[:, 0, :, :]
    for i in range(num_parameters):
        fig, ax = heatplot(densities_0[i, ...], mesh_time, mesh_space, densities_0.min(), densities_0.max())
        plt.title(r'Heat plot of density with $\mu =$' + f'{mean_reversion_speeds[i]}')
        plt.show()

    densities = torch.maximum(densities, torch.tensor(0.))
    densities_1 = densities[:, k, :, :]
    for i in range(num_parameters):
        fig, ax = heatplot(densities_1[i, ...], mesh_time, mesh_space, densities_1.min(), densities_1.max())
        plt.title(r'Heat plot of density with $\mu =$' + f'{mean_reversion_speeds[i]}')
        plt.show()

    x_min = -0.5
    x_max = 0.5
    controls, mesh_space_section = section(controls, mesh_space, x_min, x_max)
    for i in range(num_parameters):
        fig, ax = heatplot(controls[i, :, :], mesh_time, mesh_space_section, controls.min(), controls.max(),
                           x_min=x_min,
                           x_max=x_max)
        plt.title(r'Heat plot of control with $\mu =$' + f' {mean_reversion_speeds[i]}')
        plt.show()

    controls_1, mesh_space = section(controls_1, mesh_space, x_min, x_max)
    for i in range(num_parameters):
        fig, ax = heatplot(controls_1[i, :, :], mesh_time, mesh_space_section, controls_1.min(), controls_1.max(),
                           x_min=x_min,
                           x_max=x_max)
        plt.title(r'Heat plot of control with $\mu =$' + f' {mean_reversion_speeds[i]}')
        plt.show()

