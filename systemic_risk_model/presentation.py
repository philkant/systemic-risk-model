import torch
import matplotlib as mpl
from matplotlib import rc, rcParams
from matplotlib import pyplot as plt
import utilities as ut

# mpl.rcParams['mathtext.fontset'] = 'cm'
# mpl.rcParams["font.family"] = "cm"
# # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# # rc('text', usetex=True)
# rc('text', usetex=True)
# rc('axes', linewidth=2)
# rc('font', weight='bold')
# rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

torch.manual_seed(3)

n = 500

time = torch.linspace(0., 1., n + 1)

bm_inc = torch.torch.randn(n)/n**(1/2)
bm = torch.zeros(n + 1)
bm[1:] = bm_inc.cumsum(0)
bm = bm + 0.1

parameter = 10.

intensity = parameter*torch.maximum(-bm, torch.zeros(1))
cumulative_inc = (time[1:] - time[:-1])*intensity[:-1]
cumulative = torch.zeros(n + 1)
cumulative[1:] = cumulative_inc.cumsum(0)


def convert_floats_to_strings(float_list, decimal_places=2):
    string_list = [f"{round(num, decimal_places):.{decimal_places}f}" for num in float_list]
    return string_list


if __name__ == '__main__':
    # _, ax = plt.subplots()
    # ax.plot(time, bm)
    # ax.set_xlabel('Time')
    # ax.set_ylabel(r'Distance-to-breach $X^i_t$')
    # plt.show()
    #
    # _, ax = plt.subplots()
    # ax.plot(time, intensity)
    # ax.set_xlabel('Time')
    # ax.set_ylabel(r'Default intensity $\lambda (X^i_t)_-$')
    # plt.show()
    #
    # _, ax = plt.subplots()
    # ax.plot(time, cumulative)
    # ax.set_xlabel('Time')
    # ax.set_ylabel(r'Cumulative intensity $\Lambda^i_t$')
    #
    # x_ticks = convert_floats_to_strings(list(ax.get_xticks())[1:-1], 1)
    # y_ticks = convert_floats_to_strings(list(ax.get_yticks())[1:-1], 2)
    #
    # ax.set_xticks(list(ax.get_xticks())[1:-1] + [0.7205])
    # ax.set_xticklabels(x_ticks + [r'$\tau_i$'])
    #
    # ax.set_yticks(list(ax.get_yticks())[1:-1] + [0.085])
    # ax.set_yticklabels(y_ticks + [r'$\theta_i$'])
    #
    # ax.axvline(x=0.7205, ymin=0.04, ymax=0.79, linestyle='dashed', color='black', linewidth=0.5)
    # ax.axhline(y=0.085, xmin=0.04, xmax=0.70, linestyle='dashed', color='black', linewidth=0.5)
    # plt.show()

    # Plots for reduced-form model.
    fig, ax = plt.subplots(3, sharex=True)

    ax[0].plot(time, bm)
    # ax[0].set_xlabel('Time')
    ax[0].set_ylabel(r'$X^i_t$')

    ax[1].plot(time, intensity)
    # ax[1].set_xlabel('Time')
    ax[1].set_ylabel(r'$\lambda (X^i_t)_-$')

    y_ticks = convert_floats_to_strings(list(ax[1].get_yticks())[:-1], 1)
    ax[1].set_yticklabels(y_ticks)

    ax[2].plot(time, cumulative)
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel(r'$\Lambda^i_t$')

    x_ticks = convert_floats_to_strings(list(ax[2].get_xticks())[1:-1], 1)
    y_ticks = convert_floats_to_strings(list(ax[2].get_yticks())[1:-1], 1)

    ax[2].set_xticks(list(ax[2].get_xticks())[1:-1] + [0.7205])
    ax[2].set_xticklabels(x_ticks + [r'$\tau_i$'])

    ax[2].set_yticks(list(ax[2].get_yticks())[1:-1] + [0.26])
    ax[2].set_yticklabels(y_ticks + [r'$\theta_i$'])

    ax[2].axvline(x=0.7205, ymin=0.04, ymax=0.8, linestyle='dashed', color='black', linewidth=0.5)
    ax[2].axhline(y=0.26, xmin=0.04, xmax=0.70, linestyle='dashed', color='black', linewidth=0.5)

    fig.align_ylabels(ax)

    # plt.suptitle('Default time')

    plt.show()

    # Plots for structural model.
    _, ax = plt.subplots()

    ax.plot(time, bm)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$X^i_t$')

    x_ticks = convert_floats_to_strings(list(ax.get_xticks())[1:-1], 1)

    ax.set_xticks(list(ax.get_xticks())[1:-1] + [0.33])
    ax.set_xticklabels(x_ticks + [r'$\bar{\tau}_i$'])

    ax.axvline(x=0.33, ymin=0.04, ymax=0.44, linestyle='dashed', color='black', linewidth=0.5)
    ax.axhline(y=0.0, xmin=0.04, xmax=0.343, linestyle='dashed', color='black', linewidth=0.5)

    # plt.title('Default time in structural model')
    plt.show()

    # Plots of initial distribution.
    _, ax = plt.subplots()

    mesh = torch.linspace(-1., 1., 1000)
    ax.plot(mesh, ut.gamma(mesh, shape=6., rate=60.))

    ax.set_xlabel('Distance-to-breach')
    ax.set_ylabel('Density')

    # plt.title('Default time in structural model')
    plt.show()
