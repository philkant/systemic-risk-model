# This file contains all commands to run the numerical experiments for the paper "Optimal Control of McKean--Vlasov
# SDEs with Contagion Through Exponential Killing".



parallel --ungroup --colsep ' ' -j4 -a ./systemic_risk_model/configs/experiments/feedback.txt python -m systemic_risk_model.main