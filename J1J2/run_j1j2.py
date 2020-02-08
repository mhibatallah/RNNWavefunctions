from Training_J1J2 import run_J1J2

#numsteps = number of training iterations
#systemsize = the number of physical spins
#J1_, J2_ = the coupling parameters of the J1-J2 Hamiltonian
#numsamples = number of samples used for training
#num_units = number of memory units of the hidden state of the RNN
#num_layers = number of vertically stacked RNN cells
run_J1J2(numsteps = 5*10**3, systemsize = 20, J1_  = 1.0, J2_ = 0.0, num_units = 50, num_layers = 1, numsamples = 500, learningrate = 2.5*1e-3, seed = 111)
