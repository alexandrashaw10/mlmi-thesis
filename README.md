# Heterogeneous Multi-Agent Learning under Policy Constraints

This repository contains the code for the dissertation "Evaluating the Benefits of Heterogeneity in Multi-Agent Reinforcement Learning" submitted for the MPhil in Machine Learning and Machine Intelligence at the University of Cambridge Department of Engineering. Many thanks to my supervisors Prof. Amanda Prorok and Matteo Bettini.

This repository is based off of code written for:
- [Heterogeneous Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2301.07137)
- [System Neural Diversity: Measuring Behavioral Heterogeneity in Multi-Agent Learning](https://arxiv.org/abs/2305.02128)
- [VMAS: Vectorized Multi-Agent Simulator](https://arxiv.org/abs/2207.03530)

This repository also utilizes the following package:
- [Monotonenorm](https://github.com/niklasnolte/MonotonicNetworks)

### Repository Overview

#### Model

The LipNormedMultiAgentMLP model is in `models/lip_multiagent_mlp.py`. This model is used to create the multi-agent policies with Lipschitz normalization that can be controlled as a hyperparameter.

#### Experiments

The training scripts to run experiments in the various [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) scenarios can be found in the `train_torchRL` folder.

To run experiments, look for the run file for the specific scenario (e.g `run_left_right.py`).

The hyperparameters and scenario parameters can be set through the command line arguments for each run file.
