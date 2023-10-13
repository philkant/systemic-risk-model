from matplotlib import pyplot as plt
import logging
import hydra
import torch
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from systemic_risk_model import finite_element as fd, controls as ct, utilities as ut, analysis as an

# TODO: remove initialise_parameter method from fin_diff_scheme, then remove parameters from get_state method

torch.set_default_dtype(torch.float64)


shape = 6.
rate = 60.


def global_setup(seed, final_time, time_steps, min_space, max_space, space_steps, weight, control_type, hidden_size,
                 control_function, hidden_size_scalar, intensity_parameter, intensity_type, **kwargs):
    torch.manual_seed(seed)
    meshs_time = []
    meshs_space = []
    initial_conditions = []

    for i in range(len(time_steps)):
        meshs_time += [torch.linspace(0., final_time, time_steps[i] + 1)]
        endpoints = torch.tensor([min_space, max_space])
        num_points = torch.tensor([space_steps[i]])
        meshs_space += [ut.return_mesh_space(endpoints, num_points)]

        initial_condition = ut.gamma(meshs_space[i][1:-1], shape, rate)
        initial_conditions += [initial_condition/torch.dot(initial_condition,
                                                           meshs_space[i][2:] - meshs_space[i][1:-1])]

    def terminal_cost(x: torch.tensor) -> torch.tensor:
        # y = torch.maximum(x, torch.tensor(0.))
        # y = torch.minimum(10*y, torch.tensor(1.))
        # return -y
        return -torch.ones(x.size())

    if control_function == 'intensity':
        control = ct.get_control(control_type, hidden_size, fd.get_intensity(intensity_parameter, intensity_type),
                                 hidden_size_scalar)
    else:
        control = ct.get_control(control_type, hidden_size, lambda x: -torch.ones(x.size()), hidden_size_scalar)

    return meshs_time, meshs_space, lambda x: weight*x.abs(), terminal_cost, control, initial_conditions


def save_output(fin_diff_scheme_state, cost_training, density, cost, output_path):
    output = {
        "fin_diff_scheme_state": fin_diff_scheme_state,
        "cost_training": cost_training,
        "density": density,
        "cost": cost
    }

    if not os.path.exists("./data"):
        os.mkdir("./data")

    torch.save(output, output_path)
    logging.info(f"Saved output to {output_path}")


def load_output(output_path):
    output = torch.load(output_path)
    logging.info(f"Loaded output from {output_path}")
    return output


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(args):
    meshs_time, meshs_space, running_cost, terminal_cost, control, initial_conditions = global_setup(**args)

    if args.analyse:
        output = load_output(args.output_path)
        fin_diff_scheme = fd.FinDiffScheme.load_state(output["fin_diff_scheme_state"], control, running_cost,
                                                      terminal_cost)
        discretisations = []
        for i in range(len(meshs_time)):
            discretisations += [meshs_time[i].size(0) - 1]

        logging.info("Loaded finite difference scheme.")
        an.analysis(args.experiment_name, fin_diff_scheme, meshs_time[-1], meshs_space[-1], args.num_samples[-1],
                    args.num_epochs, discretisations)
    else:
        # Initialise parameters for finite difference scheme.
        fin_diff_scheme = fd.FinElteScheme(control, running_cost, terminal_cost)
        fin_diff_scheme.initialise_parameters(args.volatility, args.volatility_0, args.feedback,
                                              args.intensity_parameter, args.intensity_type, args.mean_reversion)

        logging.info("Started cloning bang bang control.")

        regularisation = 0.05

        def bangbang(x: torch.tensor) -> torch.tensor:
            y = -x/(2*regularisation) + 1/2
            y = torch.minimum(torch.maximum(y, torch.tensor(0.)), torch.tensor(1.))

            return y

        fin_diff_scheme.clone(meshs_time[0], meshs_space[0], initial_conditions[0], bangbang,
                              args.num_samples[0], 1, 1, 101, print_frequency=1, plot=False)

        logging.info("Finished cloning bang bang control.")

        # Train control and return cost for trained control.
        logging.info("Started training control.")
        cost_training = fin_diff_scheme.multilevel_fit(
            meshs_time[:-1], meshs_space[:-1], initial_conditions[:-1], args.num_samples[:-1], args.num_epochs).detach()

        logging.info("Finished training control.")

        logging.info("Started simulating control problem.")
        _, cost, _ = fin_diff_scheme(meshs_time[-2], meshs_space[-2], initial_conditions[-2],
                                     common_noise=args.num_samples[-2])

        density, _, _ = fin_diff_scheme(meshs_time[-1], meshs_space[-1], initial_conditions[-1],
                                        common_noise=args.num_samples[-1])

        logging.info("Finished simulating control problem.")
        density = density.detach()
        cost = cost.detach()

        save_output(fin_diff_scheme.get_state(), cost_training, density, cost, args.output_path)


if __name__ == '__main__':
    main()
