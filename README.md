# bc-PINN
A novel sequential method to train physics informed neural networks for Allen Cahn and Cahn Hilliard equations

Abstract:


A physics informed neural network (PINN) incorporates the physics of a system by satisfying its boundary value problem through a neural network’s loss function. The PINN approach has shown great success in approximating the map between the solution of a partial differential equation (PDE) and its spatio-temporal coordinates. However, the PINN’s accuracy suffers significantly for strongly non-linear and higher-order time-varying partial differential equations such as Allen Cahn and Cahn Hilliard equations. To resolve this problem, a novel PINN scheme is proposed that solves the PDE sequentially over successive time segments using a single neural network. The key idea is to re-train the same neural network for solving the PDE over successive time segments while satisfying the already obtained solution for all previous time segments. Thus it is named as backward compatible PINN (bc-PINN). To illustrate the advantages of bc-PINN, the Cahn Hilliard and Allen Cahn equations are solved. These equations are widely used to describe phase separation and reaction–diffusion systems. Additionally, two new techniques have been introduced to improve the proposed bc-PINN scheme. The first technique uses the initial condition of a time-segment to guide the neural network map closer to the true map over that segment. The second technique is a transfer learning approach where the features learned from the previous training are preserved. We have demonstrated that these two techniques improve the accuracy and efficiency of the bc-PINN scheme significantly. It has also been demonstrated that the convergence is improved by using a phase space representation for higher-order PDEs. It is shown that the proposed bc-PINN technique is significantly more accurate and efficient than PINN.


DOI : https://doi.org/10.1016/j.cma.2021.114474

Cite: 

BIBTEX

@article{MATTEY2022114474,

title = {A novel sequential method to train physics informed neural networks for Allen Cahn and Cahn Hilliard equations},

journal = {Computer Methods in Applied Mechanics and Engineering},

volume = {390},

pages = {114474},

year = {2022},

issn = {0045-7825},

doi = {https://doi.org/10.1016/j.cma.2021.114474},

url = {https://www.sciencedirect.com/science/article/pii/S0045782521006939},

author = {Revanth Mattey and Susanta Ghosh},

TEXT

Revanth Mattey, Susanta Ghosh,

A novel sequential method to train physics informed neural networks for Allen Cahn and Cahn Hilliard equations,

Computer Methods in Applied Mechanics and Engineering,

Volume 390,

2022,

114474,

ISSN 0045-7825,

https://doi.org/10.1016/j.cma.2021.114474.

(https://www.sciencedirect.com/science/article/pii/S0045782521006939)
