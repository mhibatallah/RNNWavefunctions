# RNN Wavefunctions

RNN Wavefunctions are efficient quantum many-body wavefunction ansätzes based on Recurrent Neural Networks. These wavefunction can be used to find the ground state of a quantum many-body Hamiltonian using Variational Monte Carlo (VMC). In our recent paper (**arxiv link**), we show that this novel architecture can provide accurate estimations of ground state energies, correlation functions as well as entanglement entropies.

Our implementation is based on TensorFlow 1 and we plan to support TensorFlow 2 and PyTorch in the future.

## Running Variational Monte Carlo (VMC) Calculations

Currently, this repository contains four folders, each one is specific for a given model and architecuture in the following order:
- **1DTFIM**: 1D Positive Recurrent Neural Network Wavefunction for 1D Transverse-field Ising Model (TFIM).
- **2DTFIM_1DRNN**: 1D Positive Recurrent Neural Network Wavefunction for 2D TFIM.
- **2DTFIM_2DRNN**: 2D Positive Recurrent Neural Network Wavefunction for 2D TFIM.
- **J1J2**: 1D Complex Recurrent Neural Network Wavefunction for 1D J1-J2 Model. PS: if you want to apply an initial Marshall sign on top of the cRNN wavefunction: go to file "Training_J1J2.py", and change **Marshall_sign** argument in the function "J1J2MatrixElements" to **True**.

We also plan to support more models and architectures in the future.

To run a VMC calculation for the task of finding the ground state energy of a certain model, it is enough to specify the following parameters in the python run file in a folder of interest.

For further questions or inquiries, please feel free to send an email to mohamed.hibat.allah@uwaterloo.ca, we would also appreciate future contributions.
