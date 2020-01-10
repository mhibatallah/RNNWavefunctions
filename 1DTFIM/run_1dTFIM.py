from RNN_wavefunction_TFIM_GRU import run_1DTFIM

run_1DTFIM(numsteps = 10**3, systemsize = 20, num_units = 50, num_layers = 1, num_samples = 500, learningrate = 5e-3, seed = 111)
