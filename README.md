# Recurrent Neural Network Wave Functions

RNN wave functions are efficient quantum many-body wave function ansätzes based on Recurrent Neural Networks. These wave functions can be used to find the ground state of a quantum many-body Hamiltonian using Variational Monte Carlo (VMC). <a href="https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023358" target="_blank">In our paper</a>, we show that this novel architecture can provide accurate estimations of ground state energies, correlation functions as well as entanglement entropies.

Our implementation is based on TensorFlow 1.

## Running Variational Monte Carlo (VMC) Calculations

This repository contains the following folders:

* **1DTFIM**: an implementation of the 1D Positive Recurrent Neural Network (pRNN) Wave Function for the purpose of finding the ground state of the 1D Transverse-field Ferromagnetic Ising Model (TFIM).

* **2DTFIM_1DRNN**: an implementation of the 1D Positive Recurrent Neural Network Wave Function for the goal of finding the ground state of the 2D TFIM.

* **2DTFIM_2DRNN**: an implementation of the 2D Positive Recurrent Neural Network Wave Function for the purpose of finding the ground state of the 2D TFIM.

* **J1J2**: an implementation of the 1D Complex Recurrent Neural Network (cRNN) Wave Function to estimate the ground state of the 1D J1-J2 model, with a built-in U(1) symmetry to impose a zero magnetization as explained in the Appendix of our paper.

* **Tutorials**: this folder contains jupyter notebooks that you can run on <a href="http://colab.research.google.com" target="_blank">Google Colaboratory</a> (with a free GPU!) to test pRNN wave functions on 1DTFIM and cRNN wave functions on 1D J1J2, and to compare with exact diagonalization for small system sizes. These notebooks will also help you to get a clearer idea on how to use the remaining code in the previous folders for further investigations.

* **Check_Points**: this folder is intended to contain the saved parameters of the RNN wave function as well as the energies and the variances after training.

We plan to support more models and architectures in the future.

To learn more about this new approach, you can check out our paper on Physical Review Research: https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.023358

For further questions or inquiries, please feel free to send an email to mohamed.hibat.allah@uwaterloo.ca. Future contributions would be really appreciated.

### Citation:
If you would like to cite this work, you can use the bibtex code below:
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
