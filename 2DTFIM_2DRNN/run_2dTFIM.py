from Training_2DTFIM_2DRNN import run_2DTFIM

#numsteps = number of training iterations
#systemsize_x = the size of the x-dimension of the square lattice
#systemsize_x = the size of the y-dimension of the square lattice
#Bx = transverse magnetic field
#numsamples = number of samples used for training
#num_units = number of memory units of the hidden state of the RNN
#num_layers is not supported yet, stay tuned!
run_2DTFIM(numsteps = 2*10**4, systemsize_x = 4, systemsize_y = 4, Bx = 3, num_units = 50, num_samples = 500, learningrate = 5e-3, seed = 111)
