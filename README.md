This respository provides code accompanying the paper  [Off-policy evaluation beyond overlap: partial identification through smoothness](https://arxiv.org/abs/2305.11812). The files and folders it contains are:

- _ope_estimators.py_ contains implementations of the methods from the paper. The main classes are `ManskiEstimator`, `LipImputeEstimator`, and `LipImputeBddRespEstimator`, which compute partial identification intervals for the off-policy value under boundedness, smoothness, and both boundednness and smoothness assumptions, respectively. 

- _yeast_, which contains code to replicate the experiments in Section 5 on the yeast dataset

- _yahoo_, which contains code to replicate the experiments in Section 5 on the Yahoo! Webscope Dataset


