# DSC531-NRT-Project
Exploration of Simulation-based inference for Model Fitting
--------
Computational models are widely used in scientific research (e.g. computational neuroscience) to explore the process of an agent (e.g. the brain). Inferring the parameters of model is critical for evaluating the model itself as well as drawing more scientific conclusion from parameters. However, traditional Bayesian inference fails on complex models due to the intractable data likelihood. Many inference methods have been proposed to account for this problem, such as sampling based methods and density estimation methods. One of the state-of-art density estimation methods is Automatic Posterior Transformation (APT). Here, we aim to use APT to fit a complex computational neuroscience model. Before that we tested APT on toy models and compared it with sampling based method, which serves as the benchmark. We found APT requires large number of simulations, sensitive to initialization and highly depends on summary statistics of observation.

The Repository contains two folders for the toy problems: 
1. Line Fitting: The main file `LineFitting.py` when run as a script calculates the analytical posterior, MH-Sampling estimate of the posterior and the APT posterior. The setting
2. Psychometric:

`
