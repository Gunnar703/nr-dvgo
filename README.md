# Non-Rigid Direct Voxel Grid Optimization
This project is a modification of https://github.com/sunset1995/DirectVoxGO/tree/main and therefore falls under the same [LICENSE](LICENSE)

## Goal
The goal is to use images of a structure (beam in this case) in its equilibrium state, as well as a video of it in free vibration, to determine the structure's material properties (Young's modulus, density, Poisson's ratio).

## Methodology
1. Train a voxel grid using the DVGO algorithm to represent the equilibrium scene. This can be accomplished by running `train_and_viz.bat`.
2. Implement a ray bending network conditioned on time and space, such that
```math 
(u, v, w) = MLP(x, y, z, t) 
```
```math 
(\sigma, \mathbf{c}) = \mathcal{G}(x - u, y - v, z - w)
```
3. Freeze the parameters of the grid $\mathcal{G}$ representing the equilibrium scene. Train the ray bending MLP such that, in conjunction with the equilibrium grid, it can reproduce the video of the structure under free vibration.

So long as the structure's deflection $\mathbf{\delta}$ depends only on coordinates $\mathbf{x}$ such that $\mathbf{\delta}\cdot\mathbf{x}=0$ (as is the case for beams and plates), it may be regularized by the physics of the structure. For example, a beam in free vibration is governed by
```math
EI\frac{\partial^4v}{\partial x^4} + \rho A\frac{\partial^2 v}{\partial t^2} = 0
```

Once the MLP is trained, the material parameters ($E$, $\rho$) can be obtained by minimizing
```math
\int\int \left( EI\frac{\partial^4v}{\partial x_3^4} + \rho A\frac{\partial^2 v}{\partial t^2} \right)^2 d\mathbf{x}dt
```
assuming the beam is oriented vertically.

## Status
Currently, the ray bending network is implemented and the equilibrium scene can be represented. A training loop to perform step 3 is implemented in `main.py`. However, the ray-bending MLP does not currently converge to an acceptable extent.