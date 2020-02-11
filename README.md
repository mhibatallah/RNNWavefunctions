# Recurrent Neural Network Wavefunctions

RNN Wavefunctions are efficient quantum many-body wavefunction ansätzes based on Recurrent Neural Networks. These wavefunction can be used to find the ground state of a quantum many-body Hamiltonian using Variational Monte Carlo (VMC). <a href="https://arxiv.org/abs/2002.02973" target="_blank">In our paper</a>, we show that this novel architecture can provide accurate estimations of ground state energies, correlation functions as well as entanglement entropies.

Our implementation is based on TensorFlow 1 and we plan to support TensorFlow 2 and PyTorch in the future.

## Running Variational Monte Carlo (VMC) Calculations

This repository contains the following folders:

* **1DTFIM**: 1D Positive Recurrent Neural Network (pRNN) Wavefunction for 1D Transverse-field Ferromagnetic Ising Model (TFIM).

* **2DTFIM_1DRNN**: 1D Positive Recurrent Neural Network Wavefunction for 2D TFIM.

* **2DTFIM_2DRNN**: 2D Positive Recurrent Neural Network Wavefunction for 2D TFIM.

* **J1J2**: 1D Complex Recurrent Neural Network (cRNN) Wavefunction (with a built-in U(1) symmetry to impose a zero magnetization) for 1D J1-J2 Model.

* **Tutorials**: this folder contains jupyter notebooks that you can run on <a href="http://colab.research.google.com" target="_blank">Google Colaboratory</a> (with a free GPU!) to test pRNN wavefunctions on 1DTFIM and cRNN wavefunctions on 1D J1J2, and to compare with exact diagonalization for small system sizes. These notebooks will also help you to get a clearer idea on how to use the remaining code in the previous folders for further investigations.

* **Check_Points**: this folder is intended to contain the saved parameters of the RNN wavefunction as well as the energies and the variances after training.

We plan to support more models and architectures in the future.

To learn more about this new approach, you can check out our paper on arxiv: https://arxiv.org/abs/2002.02973

For further questions or inquiries, please feel free to send an email to mohamed.hibat.allah@uwaterloo.ca. Future contributions would be really appreciated.

### Citation:
If you would like to cite this work, you can use the bibtex code below:
```bibtex
@misc{RNNWF2020,
    title={Recurrent Neural Network Wavefunctions},
    author={Mohamed Hibat-Allah and Martin Ganahl and Lauren E. Hayward and Roger G. Melko and Juan Carrasquilla},
    year={2020},
    eprint={2002.02973},
    archivePrefix={arXiv},
    primaryClass={cond-mat.dis-nn}
}
```
