import tensorflow as tf
import numpy as np
import random
from tensorflow.python.client import device_lib

def get_numavailable_gpus():
    local_device_protos = device_lib.list_local_devices()
    num_gpus = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
    # print("num_gpus =", num_gpus)
    return num_gpus

def binary_to_intarray(binary_samples):
    "This function transforms a binary array of size (numsamples, Nx,Ny, 3) to (_,numsamples, Nx,Ny) where integers goes between 0 and 7"
    # print(tf.shape(binary_samples))
    # numsamples,Nx,Ny,bin_length = tf.shape(binary_samples)
    prod_window = np.array([2**i for i in range(3)], dtype = np.int64)
    with tf.device('/CPU:0'): #necessary otherwise you might get a tensorflow error
        return tf.tensordot(binary_samples, prod_window, axes = [3,0])

def intarray_to_binary(samples, bin_length = 3):
    "This function does the inverse of binary_to_intarray"
    numsamples= tf.shape(samples)
    ones = tf.ones((numsamples), dtype = tf.int64)

    samples_new = tf.identity(samples)
    samples_binary_1 = tf.floormod(samples_new, 2*ones)
    samples_new = tf.floordiv(samples_new, 2*ones)

    samples_binary_2 = tf.floormod(samples_new, 2*ones)
    samples_new = tf.floordiv(samples_new, 2*ones)

    samples_binary_3 = tf.floormod(samples_new, 2*ones)

    return tf.stack([samples_binary_1, samples_binary_2, samples_binary_3], axis = 1)


class MDRNNWavefunction(object):
    def __init__(self,systemsize_x, systemsize_y,cell=None,activation=tf.nn.relu,units=[10],inputdim = 2, scope='RNNwavefunction',seed = 111):
        """
            systemsize_x:  int
                         size of x-dim
            systemsize_y: int
                          size of y_dim
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
            inputdim     size of the one hot encoding of the inputs
            scope:       str
                         the name of the name-space scope
            activation:  activation for the RNN cell
            seed:        pseudo random generator
        """
        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN wavefunction
        self.Nx=systemsize_x #number of sites in the 2d model
        self.Ny=systemsize_y
        self.N = self.Nx*self.Ny

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        #Defining the neural network
        with self.graph.as_default():
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):

              tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator
              self.rnn=cell(num_units = units[0],num_in = inputdim,name="rnn_"+str(0),dtype=tf.float64)
              self.dense = tf.compat.v1.layers.Dense(8,activation=tf.nn.softmax,name='wf_dense', dtype = tf.float64)


################################################################################

    def sample(self,numsamples,inputdim):
        """
            generate samples from a probability distribution parametrized by a recurrent neural network
            ------------------------------------------------------------------------
            Parameters:

            numsamples:      int
                             number of samples to be produced
            inputdim:        int
                             hilbert space dimension

            ------------------------------------------------------------------------
            Returns:         a tensor

            samples:         tf.Tensor of shape (numsamples,systemsize)
                             the samples in integer encoding
        """

        with self.graph.as_default(): #Call the default graph, used if willing to create multiple graphs.
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):

                #Initial input to feed to the lstm

                self.inputdim=inputdim
                self.outputdim=self.inputdim
                self.numsamples=numsamples


                samples=[[[] for ny in range(self.Ny)] for nx in range(self.Nx)]
                rnn_states = {}
                inputs = {}

                for ny in range(-2,self.Ny): #Loop over the number of sites
                    for nx in range(-2,self.Nx+2):
                        rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float64)
                        inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,3*inputdim), dtype = tf.float64) #Feed the table b in tf.

                #Begin Sampling
                for ny in range(self.Ny):

                    if ny%2 == 0:

                        for nx in range(self.Nx): #left to right

                            local_inputs = [inputs[str(nx-1)+str(ny)],inputs[str(nx)+str(ny-1)], inputs[str((nx+1)%self.Nx)+str(ny)],inputs[str(nx)+str((ny+1)%self.Ny)]]

                            local_states = [rnn_states[str(nx-1)+str(ny)], rnn_states[str(nx)+str(ny-1)], rnn_states[str((nx+1)%self.Nx)+str(ny)], rnn_states[str(nx)+str((ny+1)%self.Ny)]]

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn(local_inputs, local_states)

                            output=self.dense(rnn_output)

                            with tf.device('/CPU:0'): #necessary otherwise you might get a tensorflow error
                                sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1,])

                            samples[nx][ny] = intarray_to_binary(sample_temp)
                            inputs[str(nx)+str(ny)]=tf.reshape(tf.one_hot(samples[nx][ny],depth=self.outputdim, dtype = tf.float64), shape = [self.numsamples, 3*inputdim])


                    if ny%2 == 1:

                        for nx in range(self.Nx-1,-1,-1): #right to left

                            local_inputs = [inputs[str(nx+1)+str(ny)],inputs[str(nx)+str(ny-1)], inputs[str((nx-1)%self.Nx)+str(ny)],inputs[str(nx)+str((ny+1)%self.Ny)]]

                            local_states = [rnn_states[str(nx+1)+str(ny)], rnn_states[str(nx)+str(ny-1)], rnn_states[str((nx-1)%self.Nx)+str(ny)], rnn_states[str(nx)+str((ny+1)%self.Ny)]]

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn(local_inputs, local_states)

                            output=self.dense(rnn_output)

                            with tf.device('/CPU:0'):
                                sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1,])

                            samples[nx][ny] = intarray_to_binary(sample_temp)
                            inputs[str(nx)+str(ny)]=tf.reshape(tf.one_hot(samples[nx][ny],depth=self.outputdim, dtype = tf.float64), shape = [self.numsamples, 3*inputdim])


        self.samples=tf.transpose(a=tf.stack(values=samples,axis=0), perm = [2,0,1,3])

        return self.samples

#############################################################################################

    def log_probability(self,samples,inputdim):
        """
            calculate the log-amplitudes of ```samples``
            ------------------------------------------------------------------------
            Parameters:

            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,system-size)
                             containing the input samples in integer encoding
            inputdim:        int
                             dimension of the input space

            ------------------------------------------------------------------------
            Returns:
            log-probs      tf.Tensor of shape (number of samples,)
                             the log-probability of each sample
            """
        with self.graph.as_default():

            self.inputdim=inputdim
            self.outputdim=self.inputdim

            self.numsamples=tf.shape(samples)[0]

            #Initial input to feed to the lstm
            self.outputdim=self.inputdim

            samples_=tf.transpose(a=samples, perm = [1,2,0,3])
            rnn_states = {}
            inputs = {}

            for ny in range(-2,self.Ny): #Loop over the number of sites
                for nx in range(-2,self.Nx+2):
                    rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float64)
                    inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,3*inputdim), dtype = tf.float64) #Feed the table b in tf.

            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                probs = [[[] for ny in range(self.Ny)] for nx in range(self.Nx)]

                #Begin estimation of log amplitudes
                for ny in range(self.Ny):

                    if ny%2 == 0:

                        for nx in range(self.Nx): #left to right

                            local_inputs = [inputs[str(nx-1)+str(ny)],inputs[str(nx)+str(ny-1)], inputs[str((nx+1)%self.Nx)+str(ny)],inputs[str(nx)+str((ny+1)%self.Ny)]]

                            local_states = [rnn_states[str(nx-1)+str(ny)], rnn_states[str(nx)+str(ny-1)], rnn_states[str((nx+1)%self.Nx)+str(ny)], rnn_states[str(nx)+str((ny+1)%self.Ny)]]

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn(local_inputs, local_states)

                            output=self.dense(rnn_output)

                            probs[nx][ny] = output

                            inputs[str(nx)+str(ny)]=tf.reshape(tf.one_hot(samples_[nx,ny],depth=self.outputdim,dtype = tf.float64), shape = [self.numsamples, 3*inputdim])



                    if ny%2 == 1:

                        for nx in range(self.Nx-1,-1,-1): #right to left

                            local_inputs = [inputs[str(nx+1)+str(ny)],inputs[str(nx)+str(ny-1)], inputs[str((nx-1)%self.Nx)+str(ny)],inputs[str(nx)+str((ny+1)%self.Ny)]]

                            local_states = [rnn_states[str(nx+1)+str(ny)], rnn_states[str(nx)+str(ny-1)], rnn_states[str((nx-1)%self.Nx)+str(ny)], rnn_states[str(nx)+str((ny+1)%self.Ny)]]

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn(local_inputs, local_states)

                            output=self.dense(rnn_output)

                            probs[nx][ny] = output

                            inputs[str(nx)+str(ny)]=tf.reshape(tf.one_hot(samples_[nx,ny],depth=self.outputdim,dtype = tf.float64), shape = [self.numsamples, 3*inputdim])

            probs=tf.transpose(a=tf.stack(values=probs,axis=0),perm=[2,0,1,3])

            one_hot_samples=tf.one_hot(binary_to_intarray(samples),depth=8, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(input_tensor=tf.reduce_sum(input_tensor=tf.math.log(tf.reduce_sum(input_tensor=tf.multiply(probs,one_hot_samples),axis=3)),axis=2),axis=1)
            return self.log_probs

################ Symmetry ###################################################

    def log_probability_symmetry(self,samples,inputdim = 2,symmetry = "nosym"):

        with tf.device('/CPU:0'):

            if symmetry == "c6sym":
                list_samples = [samples]
                samples60 = rotate60(samples)
                list_samples.append(samples60)
                samples120 = rotate60(samples60)
                list_samples.append(samples120)
                samples180 = rotate60(samples120)
                list_samples.append(samples180)
                samples240 = rotate60(samples180)
                list_samples.append(samples240)
                samples300 = rotate60(samples240)
                list_samples.append(samples300)

            if symmetry == "translationsym":
                list_samples = []
                for rx in range(self.Nx):
                    for ry in range(self.Ny):
                        list_samples.append(tf.roll(samples, shift = [rx,ry], axis = [1,2]))

            group_cardinal = len(list_samples)
            numsamples = tf.shape(list_samples[0])[0]
            numgpus = get_numavailable_gpus()
            list_probs = [[] for i in range(numgpus)]
            list_samples = tf.reshape(tf.concat(list_samples, 0), [-1, self.Nx, self.Ny, 3])
            numsamplespergpu = tf.shape(list_samples)[0]//numgpus #We assume that it is divisible!

        for i in range(numgpus):
            with tf.device("/GPU:"+str(i)):
                log_prob_temp = self.log_probability(list_samples[i*numsamplespergpu:(i+1)*numsamplespergpu],inputdim)
                list_probs[i] = tf.exp(log_prob_temp)

        with tf.device('/CPU:0'):
            list_probs = tf.reshape(tf.concat(list_probs, 0), [group_cardinal,numsamples])
            return tf.math.log(tf.reduce_sum(list_probs, axis = 0)/group_cardinal)

def rotate60(samples): #in the trigonometric sens
    _,Lx, Ly, unitcellsize = samples.shape
    rotated_samples = [[[[] for j in range(unitcellsize)] for ny in range(Ly)] for nx in range(Lx)]

    for i in range(Lx):
        for j in range(Ly):
            rotated_samples[i][j][0] = samples[:,(i+j+1)%Lx,(Ly-i)%Ly,1] #up spin = 0 to left spin = 1
            rotated_samples[i][j][1] = samples[:,(i+j)%Lx,(Ly-i)%Ly,2] #left spin = 1 to right spin = 0
            rotated_samples[i][j][2] = samples[:,(i+j+1)%Lx,(Ly-(i+1))%Ly,0] #right spin = 2 to up spin = 0

    return tf.transpose(a=tf.stack(values=rotated_samples,axis=0),perm=[3,0,1,2])
