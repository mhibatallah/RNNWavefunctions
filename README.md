# Recurrent Neural Network Wave Functions

RNN wave functions are efficient quantum many-body wave function ansätzes based on Recurrent Neural Networks. These wave functions can be used to find the ground state of a quantum many-body Hamiltonian using Variational Monte Carlo (VMC). <a href="https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023358" target="_blank">In our paper</a>, we show that this architecture can provide accurate estimations of ground state energies, correlation functions as well as entanglement entropies.

In our [NeurIPS 2021 paper](https://ml4physicalsciences.github.io/2021/files/NeurIPS_ML4PS_2021_92.pdf), we also show that we can construct a tensorized version of two dimensional RNN wave functions that are capable of competing with state-of-the-art methods on the 2D Heisenberg model both on the square and the triangular lattices.

This code is an adaptation of [Martin Ganahl's code](https://github.com/mganahl/Recurrent_NN_VMC). We optimized the code, fixed some bugs, added the parity symmetry implementation, the complex RNN with U(1)-symmetry implementation along with the 2D RNN implementation.

## Dependencies
Our implementation works on Python (3.6.10) with TensorFlow (1.13.1) and NumPy (1.16.3) modules.

## Content
This repository contains the following folders:

* **1DTFIM**: an implementation of the 1D Positive Recurrent Neural Network (pRNN) Wave Function for the purpose of finding the ground state of the 1D Transverse-field Ferromagnetic Ising Model (TFIM).

* **2DTFIM_1DRNN**: an implementation of the 1D Positive Recurrent Neural Network Wave Function for the goal of finding the ground state of the 2D TFIM.

* **2DTFIM_2DRNN**: an implementation of the 2D Positive Recurrent Neural Network Wave Function for the purpose of finding the ground state of the 2D TFIM.

* **J1J2**: an implementation of the 1D Complex Recurrent Neural Network (cRNN) Wave Function to estimate the ground state of the 1D J1-J2 model, with a built-in U(1) symmetry to impose a zero magnetization as explained in the Appendix of our paper.

* **2DHeisenberg**: an implementation of the 2D Complex Recurrent Neural Network Wave Function with tensorization for the purpose of finding the ground state of the 2D Heisenberg model both in the square and the triangular lattices as described in our [NeurIPS 2021 paper](https://ml4physicalsciences.github.io/2021/files/NeurIPS_ML4PS_2021_92.pdf).

* **RydbergKagome**: an implementation of the 2D positive Recurrent Neural Network Wave Function with the gating mechanism, periodic boundary conditions, and annealing to study the ground state of the Rydberg atoms arrays on Kagome lattice.

* **Tutorials**: this folder contains jupyter notebooks that you can run on <a href="http://colab.research.google.com" target="_blank">Google Colaboratory</a> (with a free GPU!) to test pRNN wave functions on 1DTFIM and cRNN wave functions on 1D J1J2, and to compare with exact diagonalization for small system sizes. These notebooks will also help you to get a clearer idea on how to use the remaining code in the previous folders for further investigations.

* **Check_Points**: this folder is intended to contain the saved parameters of the RNN wave function as well as the energies and the variances after training.

To learn more about this approach, you can check out our paper on Physical Review Research: https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023358

You can also check our NeurIPS 2021 paper at https://ml4physicalsciences.github.io/2021/files/NeurIPS_ML4PS_2021_92.pdf.

For further questions or inquiries, please feel free to send an email to mohamed.hibat.allah@uwaterloo.ca. Future contributions would be really appreciated.

## License
The [license](https://github.com/mhibatallah/RNNWavefunctions/blob/master/LICENSE.md) of this work is derived from the BSD-3-Clause license. Ethical clauses are added to promote good uses of this code.

## Citing
```bibtex
@article{PhysRevResearch.2.023358,
  title = {Recurrent neural network wave functions},
  author = {Hibat-Allah, Mohamed and Ganahl, Martin and Hayward, Lauren E. and Melko, Roger G. and Carrasquilla, Juan},
  journal = {Phys. Rev. Research},
  volume = {2},
  issue = {2},
  pages = {023358},
  numpages = {17},
  year = {2020},
  month = {Jun},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevResearch.2.023358},
  url = {https://link.aps.org/doi/10.1103/PhysRevResearch.2.023358}
}
```
