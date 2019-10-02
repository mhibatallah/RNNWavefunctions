import tensorflow as tf
import numpy as np
import random


class RNNwavefunction(object):
    def __init__(self,systemsize_x, systemsize_y,cell=tf.contrib.rnn.LSTMCell,activation=tf.nn.relu,units=[10],scope='RNNwavefunction',seed = 111):
        """
            systemsize:  int
                         number of sites
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
            scope:       str
                         the name of the name-space scope
            homogeneous: bool
                         True: use the same RNN cell at each
                         False: use a different RNN cell at each site
            activation = tf.nn.relu (you can try using softmax function for comparaison)
        """
        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN wavefunction
        self.Nx=systemsize_x #number of sites in the 2d model
        self.Ny=systemsize_y

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        #Defining the neural network
        with self.graph.as_default():
            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
                tf.set_random_seed(seed)  # tensorflow pseudo-random generator
                self.lstm=tf.nn.rnn_cell.MultiRNNCell([cell(units[n]) for n in range(len(units))])
                # self.lstm=tf.nn.rnn_cell.MultiRNNCell([cell(units[n],activation=activation,name='LSTM_{0}'.format(n), dtype = tf.float64) for n in range(len(units))])
                self.dense = tf.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense', dtype = tf.float64)

    def sample(self,numsamples,inputdim):
        """
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            Parameters:

            numsamples:      int
                             number of samples to be produced
            inputdim:        int
                             hilbert space dimension

            ------------------------------------------------------------------------
            Returns:         a tuple (samples,log-probs)

            samples:         tf.Tensor of shape (numsamples,systemsize)
                             the samples in integer encoding
            log-probs        tf.Tensor of shape (numsamples,)
                             the log-probability of each sample
        """

        with self.graph.as_default(): #Call the default graph, used if willing to create multiple graphs.
            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
                b=np.zeros((numsamples,inputdim)).astype(np.float64)
                b[:,0]=np.ones(numsamples)
                #b = state of one spin for all the samples, this command above makes all the samples having 1 in the first component and 0 in the second.

                inputs=tf.constant(dtype=tf.float64,value=b,shape=[numsamples,inputdim]) #Feed the table b in tf.
                #Initial input to feed to the lstm

                self.inputdim=inputs.shape[1]
                self.outputdim=self.inputdim
                self.numsamples=inputs.shape[0]


                samples=[]

                lstm_state=self.lstm.zero_state(self.numsamples,dtype=tf.float64)
                #see https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/RNNCell
                #zero state returns a zero filled tensor withs shape = (self.numsamples, num_units)

                for ny in range(self.Ny): #Loop over the number of sites
                  for nx in range(self.Nx):
                    lstm_output, lstm_state = self.lstm(inputs, lstm_state)
                    output=self.dense(lstm_output)
                    sample_temp=tf.reshape(tf.multinomial(tf.log(output),num_samples=1),[-1,])
                    samples.append(sample_temp)
                    inputs=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float64)

        self.samples=tf.stack(values=samples,axis=1) # (self.N, num_samples) to (num_samples, self.N): Generate self.numsamples vectors of size self.N spin containing 0 or 1

        return self.samples

    def log_probability(self,samples,inputdim):
        """
            calculate the log-probabilities of ```samples``
            ------------------------------------------------------------------------
            Parameters:

            samples:         tf.Tensor
                             a tf.placeholder of shape (number of samples,system-size)
                             containing the input samples in integer encoding
            inputdim:        int
                             dimension of the input space

            ------------------------------------------------------------------------
            Returns:
            log-probs        tf.Tensor of shape (number of samples,)
                             the log-probability of each sample
            """
        with self.graph.as_default():

            self.inputdim=inputdim
            self.outputdim=self.inputdim

            self.numsamples=tf.shape(samples)[0]
            a=tf.ones(self.numsamples, dtype=tf.float64)
            b=tf.zeros(self.numsamples, dtype=tf.float64)

            inputs=tf.stack([a,b], axis = 1)

            with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
                probs=[]

                lstm_state=self.lstm.zero_state(self.numsamples,dtype=tf.float64)

                for ny in range(self.Ny):
                  for nx in range(self.Nx):
                      lstm_output, lstm_state = self.lstm(inputs, lstm_state)
                      output=self.dense(lstm_output)
                      probs.append(output)
                      inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(ny*self.Nx+nx)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim,dtype = tf.float64),shape=[self.numsamples,self.inputdim])

            probs=tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1])
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            #mask=tf.greater(one_hot_samples,0.001)
            #zeros = tf.zeros_like(probs)
            #self.log_probs=tf.reduce_sum(tf.log(tf.reduce_sum(tf.where(mask,probs,zeros),axis=2)),axis=1)

            self.log_probs=tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

            return self.log_probs
