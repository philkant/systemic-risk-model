import torch
from typing import Callable

from systemic_risk_model import utilities as ut

torch.set_default_dtype(torch.float64)


class Control(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t: torch.tensor, x: torch.tensor, rho: torch.tensor) -> torch.tensor:
        raise NotImplementedError()


class ConstantControl(Control):
    def __init__(self, initial_constant: float = None):
        super().__init__()
        if initial_constant is None:
            self.constant = torch.nn.Parameter(torch.rand(1).abs())
        else:
            self.constant = torch.nn.Parameter(torch.tensor(initial_constant))

    def forward(self, t: torch.tensor, x: torch.tensor, rho: torch.tensor) -> torch.tensor:
        return self.constant*torch.ones(x[1:-1].size())


class BangBangControl(Control):
    def __init__(self):
        super().__init__()
        self.upper = torch.nn.Parameter(torch.tensor(0.002543))
        self.diff = torch.nn.Parameter(torch.tensor(-3.104063))
        self.max_ctrl = torch.nn.Parameter(torch.tensor(2.486836))

    def forward(self, t: torch.tensor, y: torch.tensor, rho: torch.tensor) -> torch.tensor:
        diff = torch.sigmoid(self.diff)

        x = -y[1:-1]/diff + self.upper/diff
        x = torch.minimum(x, torch.tensor(1.))
        x = torch.maximum(x, torch.tensor(0.))

        return self.max_ctrl*x


def scalar_nonlinearity(x: torch.tensor) -> torch.tensor:
    return x*torch.sin(x)


def nonlinearity(t: torch.tensor, x: torch.tensor, y: torch.tensor):
    return torch.sin(t + x)*(x + y)/5


class NonlinearControl(Control):
    def __init__(self):
        super().__init__()

    def forward(self, t: torch.tensor, x: torch.tensor, rho: torch.tensor) -> torch.tensor:
        scalar = ut.integrate(scalar_nonlinearity(x[1:-1]), rho, x)

        return nonlinearity(t, x[1:-1], scalar.unsqueeze(-1))


class NeuralControl(Control):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.input_layer_space = torch.nn.Linear(1, hidden_size)
        self.input_layer_time = torch.nn.Linear(1, hidden_size, bias=False)
        self.hidden_layer = torch.nn.Linear(hidden_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, 1)

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, t: torch.tensor, x: torch.tensor, rho: torch.tensor) -> torch.tensor:
        # Normalise inputs.
        y = x/x.var()
        y = self.input_layer_space(y[1:-1].unsqueeze(-1)) + self.input_layer_time(t.unsqueeze(-1).unsqueeze(-1))
        y = self.leaky_relu(y)
        y = self.hidden_layer(y)
        y = self.leaky_relu(y)
        y = self.output_layer(y)

        return y.squeeze(-1)


class NeuralScalarControl(Control):
    def __init__(self, hidden_size: int, function: Callable):
        super().__init__()
        self.function = function
        self.input_layer_time = torch.nn.Linear(1, hidden_size)
        self.input_layer_space = torch.nn.Linear(1, hidden_size, bias=False)
        self.input_layer_measure = torch.nn.Linear(1, hidden_size, bias=False)
        self.hidden_layer = torch.nn.Linear(hidden_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, 1)

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, t: torch.tensor, x: torch.tensor, rho: torch.tensor) -> torch.tensor:
        scalar = ut.integrate(self.function(x[1:-1]), rho, x)/3**(1/2)

        # Normalise inputs.
        y = x/(x.var()*3**(1/2))
        s = 3*t/3**(1/2)
        y = (self.input_layer_time(s.unsqueeze(-1).unsqueeze(-1)) + self.input_layer_space(y[1:-1].unsqueeze(-1))
             + self.input_layer_measure(scalar.unsqueeze(1).unsqueeze(1)))
        y = self.leaky_relu(y)
        y = self.hidden_layer(y)
        y = self.leaky_relu(y)
        y = self.output_layer(y)

        return y.squeeze(-1)


class NeuralLearnedScalarControl(Control):
    def __init__(self, hidden_size: int, hidden_size_scalar: int):
        super().__init__()
        self.hidden_size_scalar = hidden_size_scalar
        self.input_layer_scalar = torch.nn.Linear(1, hidden_size_scalar)
        self.input_layer_scalar_time = torch.nn.Linear(1, hidden_size_scalar, bias=False)
        self.hidden_layer_scalar = torch.nn.Linear(hidden_size_scalar, hidden_size_scalar)
        # self.hidden_layer_scalar = torch.nn.Linear(hidden_size_scalar, 1)
        self.output_layer_scalar = torch.nn.Linear(hidden_size_scalar, hidden_size_scalar)

        self.input_layer_time = torch.nn.Linear(1, hidden_size)
        self.input_layer_space = torch.nn.Linear(1, hidden_size, bias=False)
        self.input_layer_measure = torch.nn.Linear(hidden_size_scalar, hidden_size, bias=False)
        # self.input_layer_measure = torch.nn.Linear(1, hidden_size, bias=False)
        self.hidden_layer = torch.nn.Linear(hidden_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, 1)

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, t: torch.tensor, x: torch.tensor, rho: torch.tensor) -> torch.tensor:
        # Normalise inputs.
        y = x/x.var()

        scalar = (self.input_layer_scalar_time(t.unsqueeze(-1).unsqueeze(-1))
                  + self.input_layer_scalar(y[1:-1].unsqueeze(-1)))
        scalar = self.leaky_relu(scalar)
        scalar = self.hidden_layer_scalar(scalar)
        scalar = self.leaky_relu(scalar)
        scalar = self.output_layer_scalar(scalar)
        # scalar = torch.matmul(rho*(x[2:] - x[1:-1]), scalar)/(3*self.hidden_size_scalar)**(1/2)
        # scalar = 5*torch.matmul(rho*(x[2:] - x[1:-1]), scalar)
        scalar = 5*((rho*(x[2:] - x[1:-1])).unsqueeze(-1)*scalar).sum(1)

        y = (self.input_layer_time(t.unsqueeze(-1).unsqueeze(-1)) + self.input_layer_space(y[1:-1].unsqueeze(-1))
             + self.input_layer_measure(scalar.unsqueeze(1)))
        y = self.leaky_relu(y)
        y = self.hidden_layer(y)
        y = self.leaky_relu(y)
        y = self.output_layer(y)

        return y.squeeze(-1)


class NeuralMeanRevertingControl(Control):
    def __init__(self, hidden_size: int, hidden_size_scalar: int, speed: float = .2):
        super().__init__()
        self.hidden_size_scalar = hidden_size_scalar
        self.input_layer_scalar = torch.nn.Linear(1, hidden_size_scalar)
        self.input_layer_scalar_time = torch.nn.Linear(1, hidden_size_scalar, bias=False)
        self.hidden_layer_scalar = torch.nn.Linear(hidden_size_scalar, hidden_size_scalar)
        # self.hidden_layer_scalar = torch.nn.Linear(hidden_size_scalar, 1)
        self.output_layer_scalar = torch.nn.Linear(hidden_size_scalar, hidden_size_scalar)

        self.input_layer_time = torch.nn.Linear(1, hidden_size)
        self.input_layer_space = torch.nn.Linear(1, hidden_size, bias=False)
        self.input_layer_measure = torch.nn.Linear(hidden_size_scalar, hidden_size, bias=False)
        # self.input_layer_measure = torch.nn.Linear(1, hidden_size, bias=False)
        self.hidden_layer = torch.nn.Linear(hidden_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, 1)

        self.speed = speed

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, t: torch.tensor, x: torch.tensor, rho: torch.tensor) -> torch.tensor:
        # Normalise inputs.
        y = x/x.var()

        scalar = (self.input_layer_scalar_time(t.unsqueeze(-1).unsqueeze(-1))
                  + self.input_layer_scalar(y[1:-1].unsqueeze(-1)))
        scalar = self.leaky_relu(scalar)
        scalar = self.hidden_layer_scalar(scalar)
        scalar = self.leaky_relu(scalar)
        scalar = self.output_layer_scalar(scalar)
        # scalar = torch.matmul(rho*(x[2:] - x[1:-1]), scalar)/(3*self.hidden_size_scalar)**(1/2)
        # scalar = 5*torch.matmul(rho*(x[2:] - x[1:-1]), scalar)
        scalar = 5*((rho*(x[2:] - x[1:-1])).unsqueeze(-1)*scalar).sum(1)

        y = (self.input_layer_time(t.unsqueeze(-1).unsqueeze(-1)) + self.input_layer_space(y[1:-1].unsqueeze(-1))
             + self.input_layer_measure(scalar.unsqueeze(1)))
        y = self.leaky_relu(y)
        y = self.hidden_layer(y)
        y = self.leaky_relu(y)
        y = self.output_layer(y).squeeze(-1)

        # Add mean-reversion.
        y = y + self.speed*(torch.matmul(rho*(x[2:] - x[1:-1]), x[1:-1]).unsqueeze(-1) - x[1:-1])

        return y.squeeze(-1)


class NeuralSemiClosedControl(Control):
    def __init__(self, input_size: int, hidden_size_coeff: int, hidden_size_readout: int):
        super().__init__()
        self.input_size = input_size
        self.input_layer_coeff = torch.nn.Linear(input_size, hidden_size_coeff)
        self.input_layer_coeff_time = torch.nn.Linear(1, hidden_size_coeff, bias=False)
        self.hidden_layer_coeff = torch.nn.Linear(hidden_size_coeff, hidden_size_coeff)
        self.output_layer_coeff = torch.nn.Linear(hidden_size_coeff, 2*input_size)

        self.input_layer_readout = torch.nn.Linear(input_size, hidden_size_readout)
        self.input_layer_readout_time = torch.nn.Linear(1, hidden_size_readout, bias=False)
        self.input_layer_readout_space = torch.nn.Linear(1, hidden_size_readout, bias=False)
        self.hidden_layer_readout = torch.nn.Linear(hidden_size_readout, hidden_size_readout)
        self.output_layer_readout = torch.nn.Linear(hidden_size_readout, 1)

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.1)

        self.initial_state = torch.nn.Linear(input_size, 1, bias=False)
        self.control_state = None

    def initialise(self, initial_state: torch.tensor = None):
        if initial_state is None:
            self.control_state = self.initial_state.weight.data
        else:
            self.control_state = initial_state.unsqueeze(0)

    def readout(self, t: torch.tensor, x: torch.tensor) -> torch.tensor:
        y = x/x.var()
        y = (self.input_layer_readout_time(t.unsqueeze(-1).unsqueeze(-1))
             + self.input_layer_readout_space(y[1:-1].unsqueeze(-1))
             + self.input_layer_readout(self.control_state.unsqueeze(1)))
        y = self.leaky_relu(y)
        y = self.hidden_layer_readout(y)
        y = self.leaky_relu(y)
        y = self.output_layer_readout(y)

        return y.squeeze(-1)

    def step(self, t: torch.tensor, time_increment: torch.tensor, brownian_increment: torch.tensor):
        if self.control_state is None:
            raise NotImplementedError('control_state has not been initialised.')

        self.control_state = self.control_state/self.input_size**(1/2)
        coeff = self.input_layer_coeff_time(t.unsqueeze(-1)) + self.input_layer_coeff(self.control_state)
        coeff = self.leaky_relu(coeff)
        coeff = self.hidden_layer_coeff(coeff)
        coeff = self.leaky_relu(coeff)
        coeff = self.output_layer_coeff(coeff)
        self.control_state = (self.control_state + coeff[:, :self.input_size]*time_increment
                              + coeff[:, self.input_size:]*brownian_increment.unsqueeze(-1))

    def forward(self, t: torch.tensor, x: torch.tensor, time_increment: torch.tensor,
                brownian_increment: torch.tensor) -> torch.tensor:
        y = self.readout(t, x)

        self.step(t, time_increment, brownian_increment)

        return y


class OrnsteinUhlenbeckDrift(Control):
    def __init__(self, mean_reversion_rate: float, mean_reversion_level: float):
        super().__init__()
        self.mean_reversion_rate = mean_reversion_rate
        self.mean_reversion_level = mean_reversion_level

    def forward(self, t: torch.tensor, x: torch.tensor, rho: torch.tensor) -> torch.tensor:
        return self.mean_reversion_rate*(self.mean_reversion_level - x[1:-1]).unsqueeze(0)


class HeatEquationDrift(Control):
    def __init__(self):
        super().__init__()

    def forward(self, t: torch.tensor, x: torch.tensor, rho: torch.tensor) -> torch.tensor:
        return torch.zeros(x[1:-1].size())


def weight_reset(layer):
    reset_parameters = getattr(layer, "reset_parameters", None)
    if callable(reset_parameters):
        layer.reset_parameters()


def get_control(control_type: str, hidden_size: int, control_function: Callable = lambda x: torch.ones(x.size()),
                hidden_size_scalar: int = 0):
    if control_type == 'NeuralLearnedScalarControl':
        control = NeuralLearnedScalarControl(hidden_size, hidden_size_scalar)
    elif control_type == 'NeuralScalarControl':
        control = NeuralScalarControl(hidden_size, control_function)
    elif control_type == 'NeuralControl':
        control = NeuralControl(hidden_size)
    elif control_type == 'NeuralSemiClosedControl':
        control = NeuralSemiClosedControl(hidden_size_scalar, hidden_size, hidden_size)
    else:
        raise ValueError(f'Control type {control_type} is not admissible.')

    return control
