from ComplexRNN_wavefunction_J1J2 import run_J1J2

run_J1J2(numsteps = 10**3, systemsize = 20, J1_  = 1.0, J2_ = 0.0, num_units = 50, num_layers = 1, num_samples = 500, learningrate = 2.5*1e-3, seed = 111)