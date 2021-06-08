# PINN-in-Pytorch

Cadet in Neural Network

Implementation of PINN from Raissi in Pytorch
Continuous Time Inference of Burgers' Equation

Existing Issues:
Loss not converging well, especially the part on boundaries.
Currently the optimizer L-BFGS-B as stated in Raissi's paper is not available. 

Discussion and Critics are welcomed.

Main Reference: 

https://maziarraissi.github.io/PINNs/

Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.

https://github.com/rodsveiga/PINNs
(Helps rewriting Raissi's code into Pytorch)
