from RNN_wavefunction_TFIM_OBC_GRU import runvmc

runvmc(numsteps = 10**3, systemsize = 20, num_units = 50, num_layers = 1, num_samples = 500, learningrate = 5e-3, seed = 111)
