# Feedback

Vary the feedback parameter $\alpha$ within the range $0.5$ to $1.5$. 
For given $n \geq 1$, choose $\alpha_i = 0.5 + (1.5 - 0.5)*i/n$ for $i = 0$, ..., $n - 1$. 
We fix $n = 3$.

## Config file (yaml)

`volatility: 0.2
volatility_0: 0.1
feedback: ???
intensity: 10.
weight: 5.`

## Argument file (txt)

`+experiments=feedback ++experiment_name="feedback" ++experiment_id=i ++feedback=\alpha_i`

## Bash command

`parallel --ungroup --colsep ' ' -j4 -a ./systemic_risk_model/configs/experiments/feedback.txt python -m systemic_risk_model.main`

# Volatility of Common Noise

Vary the volatility of the common noise $\sigma_0$ within the range $0$ to $0.5$. 
For given $n \geq 1$, choose $\sigma_0^i = 0.5*i/n$ for $i = 0$, ..., $n - 1$.
We fix $n = 10$.

## Config file (yaml)

`volatility: 0.2
volatility_0: ???
feedback: 1.
intensity: 10.
weight: 5.`

## Argument file (txt)

`+experiments=volatility ++experiment_name="volatility" ++experiment_id=i ++volatility_0=\sigma_0^i`

## Bash command

`parallel --ungroup --colsep ' ' -j4 -a ./systemic_risk_model/configs/experiments/volatility.txt python -m systemic_risk_model.main`

# Capital Requirement

Vary the capital requirement $c$ within the range $0$ to $0.2$ and fix a discount rate $d \in [0, 1]$. 
For given $n \geq 1$, choose $c_i = 0.2*i/n$ for $i = 0$, ..., $n - 1$. and set feedback  
$\alpha_i = ((d - c_i)/(1 - c_i))_+D$, where D are the innerbank liabilities. 
We fix $d = 0.2$, $D = 5.$, $n = 10$.

## Config file (yaml)

`volatility: 0.2
volatility_0: 0.1
feedback: ???
intensity: 10.
weight: 5.`

## Argument file (txt)

`+experiments=capital_requirement ++experiment_name="capital_requirement" ++experiment_id=i ++feedback=c_i`

## Bash command

`parallel --ungroup --colsep ' ' -j4 -a ./systemic_risk_model/configs/experiments/capital_requirement.txt python -m systemic_risk_model.main`