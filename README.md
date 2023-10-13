# Systemic Risk model

## Install

Developed with Python 3.9.7.

Install the Python requirements.

```bash
>> python -m pip install -r requirements.txt
```

Download GNU Parallel from https://www.gnu.org/software/parallel/ and install.

## Running numerical experiments from paper

To train the model for the feedback experiment run the following command:

```bash
>> parallel --ungroup --colsep ' ' -j4 -a ./systemic_risk_model/configs/experiments/feedback.txt python -m systemic_risk_model.main
```

The argument `-j` takes the number of cores used for parallelisation. To produce the plots run the command

```bash
>> parallel --ungroup --colsep ' ' -j4 -a ./systemic_risk_model/configs/experiments/feedback_plot.txt python -m systemic_risk_model.main
```

For the correlation and intensity experiments follow the same pattern, running the commands

```bash
>> parallel --ungroup --colsep ' ' -j4 -a ./systemic_risk_model/configs/experiments/correlation.txt python -m systemic_risk_model.main
```

```bash
>> parallel --ungroup --colsep ' ' -j4 -a ./systemic_risk_model/configs/experiments/correlation_plot.txt python -m systemic_risk_model.main
```

```bash
>> parallel --ungroup --colsep ' ' -j4 -a ./systemic_risk_model/configs/experiments/intensity.txt python -m systemic_risk_model.main
```

```bash
>> parallel --ungroup --colsep ' ' -j4 -a ./systemic_risk_model/configs/experiments/intensity_plot.txt python -m systemic_risk_model.main
```

## Running an experiment with a single set of parameters

This code uses `hydra` for configuration management. The config files are in the folder
`systemic_risk_model/configs`, with the main config `config.yaml` and the configs per experiment
in `systemic_risk_model/configs/experiment/`.

To run with the main config, call:

`python -m systematic_risk_model.main`

You can overwrite parameters in the base config. For example to change the base value of `num_epochs`:

`python -m systematic_risk_model.main +num_epochs=50`

For separate experiments that use part of the main config and partly other parameters,
make a `<experiment_name>.yaml` file in `systemic_risk_model/configs/experiment/` and run it with:

`python -m systematic_risk_model.main +experiment=<experiment_name>`

Make sure to add the line `# @package _global_` to the top of the `<experiment_name>.yaml` file.
An experiment run with `<experiment_name>.yaml` as the config now uses all the parameters specified
in there and for any unspecified parameters uses the base config file `config.yaml`.

### Hydra details

The function `main()` in `systematic_risk_model/main.py` knows which config to use
because of the decorator above the function definition:

`@hydra.main(config_path="configs", config_name="config")`

By adding `+experiment=<experiment_name>` to the python call the base config name (in this case
`config`) gets overwritten.

## Running an experiment with specified combination of parameters

Create a txt file with the desired parameter combinations

Run the script with: `parallel --colsep ' ' -a ./systemic_risk_model/configs/*.txt python -m systemic_risk_model.main`