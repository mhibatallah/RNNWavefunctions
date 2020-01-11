from RNN_wavefunction_2DTFIM_1DRNN import run_2DTFIM

run_2DTFIM(numsteps = 2*10**4, systemsize_x = 4, systemsize_y = 4, Bx = 3, num_units = 50, num_layers = 1, num_samples = 500, learningrate = 1e-3, seed = 333)
