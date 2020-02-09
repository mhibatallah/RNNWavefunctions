from TrainingRNN_1DTFIM import run_1DTFIM

#numsteps = number of training iterations
#systemsize = number of physical spins
#Bx = transverse magnetic field
#numsamples = number of samples used for training
#num_units = number of memory units of the hidden state of the RNN
#num_layers = number of vertically stacked RNN cells
RNNEnergy, varRNNEnergy = run_1DTFIM(numsteps = 10**3, systemsize = 20, Bx = +1, num_units = 50,  num_layers = 1, numsamples = 500, learningrate = 5e-3, seed = 111)

#RNNEnergy is a numpy array of the variational energy of the pRNN wavefunction
#varRNNEnergy is a numpy array of the variance of the variational energy of the pRNN wavefunction
