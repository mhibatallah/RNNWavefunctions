import tensorflow as tf
import numpy as np
import random
from tensorflow.python.client import device_lib

def get_numavailable_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

def phase_softsign(inputs):
    return np.pi*tf.nn.softsign(inputs)

def phase_tanh(inputs):
    return np.pi*tf.nn.tanh(inputs)

def phase_atan(inputs):
    return tf.atan(inputs)

def heavyside(inputs):
    sign = tf.sign(tf.sign(inputs) + 0.1 ) #tf.sign(0) = 0, this is a way to cure that since I want tf.sign(0) = 1
    return 0.5*(sign+1.0)

def regularized_identity(inputs, epsilon = 1e-4):
    sign = tf.sign(tf.sign(inputs) + 0.1 ) #tf.sign(0) = 0, this is a way to cure that since I want tf.sign(0) = 1
    return tf.stop_gradient(sign)*tf.sqrt(inputs**2 + epsilon**2)

class RNNwavefunction(object):
    def __init__(self,systemsize_x, systemsize_y,cell=None,activation=tf.nn.relu,units=[10],scope='RNNwavefunction',seed = 111, mag_fixed = True, magnetization = 0):
        """
            systemsize_x:  int
                         size of x-dim
            systemsize_y: int
                          size of y_dim
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
            scope:       str
                         the name of the name-space scope
            activation:  activation for the RNN cell
            seed:        pseudo random generator
            mag_fixed:   bool to whether fix the magnetization or not
            magnetization: value of magnetization if mag_fixed = True
        """
        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN wavefunction
        self.Nx=systemsize_x #number of sites in the 2d model
        self.Ny=systemsize_y
        self.N = self.Nx*self.Ny
        self.magnetization = magnetization
        self.mag_fixed = mag_fixed

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator
        list_in = [2]+units[:-1]

        #Defining the neural network
        with self.graph.as_default():
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):

              tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator
              self.rnn=cell(num_units = units[0],num_in = list_in[0],name="rnn_"+str(0),dtype=tf.float64)

              self.dense = tf.compat.v1.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense', dtype = tf.float64)
              # self.dense_phase = tf.compat.v1.layers.Dense(2,activation=phase_atan,name='wf_dense_phase', dtype = tf.float64)
              self.dense_phase = tf.compat.v1.layers.Dense(2,activation=phase_softsign,name='wf_dense_phase', dtype = tf.float64)

################################################################################

    def normalization(self, probs, num_up, num_generated_spins, magnetization):
        num_down = num_generated_spins - num_up
        activations_up = heavyside(((self.N+magnetization)//2-1) - num_up)
        activations_down = heavyside(((self.N-magnetization)//2-1) - num_down)

        probs = probs*tf.cast(tf.stack([activations_down,activations_up], axis = 1), tf.float64)
        probs = probs/(tf.reshape(tf.norm(tensor=probs, axis = 1, ord=1), [self.numsamples,1])) #l1 normalizing
        return probs

################################################################################

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


                samples=[[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
                rnn_states = {}
                inputs = {}

                for ny in range(-2,self.Ny): #Loop over the number of sites
                    for nx in range(-2,self.Nx+2):
                        rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float64)
                        inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.

                num_up = tf.zeros(self.numsamples, dtype = tf.float64)
                num_generated_spins = 0

                #Begin Sampling
                for ny in range(self.Ny):

                    if ny%2 == 0:

                        for nx in range(self.Nx): #left to right

                            local_inputs = [inputs[str(nx-1)+str(ny)],inputs[str(nx)+str(ny-1)]]

                            local_states = [rnn_states[str(nx-1)+str(ny)], rnn_states[str(nx)+str(ny-1)]]

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn(local_inputs, local_states)

                            output=self.dense(rnn_output)

                            if self.mag_fixed:
                                output = self.normalization(output, num_up, num_generated_spins, self.magnetization)

                            with tf.device('/CPU:0'): #necessary otherwise you might get a tensorflow error
                                sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1,])

                            samples[nx][ny] = sample_temp
                            inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float64)

                            num_generated_spins += 1
                            num_up = tf.add(num_up,tf.cast(sample_temp, tf.float64))

                    if ny%2 == 1:

                        for nx in range(self.Nx-1,-1,-1): #right to left

                            local_inputs = [inputs[str(nx+1)+str(ny)],inputs[str(nx)+str(ny-1)]]

                            local_states = [rnn_states[str(nx+1)+str(ny)], rnn_states[str(nx)+str(ny-1)]]

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn(local_inputs, local_states)

                            output=self.dense(rnn_output)
                            if self.mag_fixed:
                                output = self.normalization(output, num_up, num_generated_spins, self.magnetization)

                            with tf.device('/CPU:0'):
                                sample_temp=tf.reshape(tf.random.categorical(tf.math.log(output),num_samples=1),[-1,])

                            samples[nx][ny] = sample_temp
                            inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float64)

                            num_generated_spins += 1
                            num_up = tf.add(num_up,tf.cast(sample_temp, tf.float64))


        self.samples=tf.transpose(a=tf.stack(values=samples,axis=0), perm = [2,0,1])

        return self.samples

#############################################################################################

    def log_amplitude_nosymmetry(self,samples,inputdim):
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
            log-probs, log_phases        tf.Tensor of shape (number of samples,)
                             the log-probability of each sample and the log-phase of each sample
            """
        with self.graph.as_default():

            self.inputdim=inputdim
            self.outputdim=self.inputdim

            self.numsamples=tf.shape(input=samples)[0]

            #Initial input to feed to the lstm
            self.outputdim=self.inputdim

            samples_=tf.transpose(a=samples, perm = [1,2,0])
            rnn_states = {}
            inputs = {}

            for ny in range(-2,self.Ny): #Loop over the number of sites
                for nx in range(-2,self.Nx+2):
                    rnn_states[str(nx)+str(ny)]=self.rnn.zero_state(self.numsamples,dtype=tf.float64)
                    inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.

            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                probs = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
                log_phases = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]

                num_up = tf.zeros(self.numsamples, dtype = tf.float64)
                num_generated_spins = 0

                #Begin estimation of log amplitudes
                for ny in range(self.Ny):

                    if ny%2 == 0:

                        for nx in range(self.Nx): #left to right

                            local_inputs = [inputs[str(nx-1)+str(ny)],inputs[str(nx)+str(ny-1)]]

                            local_states = [rnn_states[str(nx-1)+str(ny)], rnn_states[str(nx)+str(ny-1)]]

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn(local_inputs, local_states)

                            output=self.dense(rnn_output)
                            output_phase = self.dense_phase(rnn_output)

                            if self.mag_fixed:
                                probs[nx][ny] = self.normalization(output, num_up, num_generated_spins, self.magnetization)
                            else:
                                probs[nx][ny] = output

                            log_phases[nx][ny] = output_phase

                            inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim,dtype = tf.float64)

                            num_generated_spins += 1
                            num_up = tf.add(num_up,tf.cast(samples_[nx,ny], tf.float64))


                    if ny%2 == 1:

                        for nx in range(self.Nx-1,-1,-1): #right to left

                            local_inputs = [inputs[str(nx+1)+str(ny)],inputs[str(nx)+str(ny-1)]]

                            local_states = [rnn_states[str(nx+1)+str(ny)], rnn_states[str(nx)+str(ny-1)]]

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn(local_inputs, local_states)

                            output=self.dense(rnn_output)
                            output_phase = self.dense_phase(rnn_output)

                            if self.mag_fixed:
                                probs[nx][ny] = self.normalization(output, num_up, num_generated_spins, self.magnetization)
                            else:
                                probs[nx][ny] = output

                            log_phases[nx][ny] = output_phase
                            inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim,dtype = tf.float64)

                            num_generated_spins += 1
                            num_up = tf.add(num_up,tf.cast(samples_[nx,ny], tf.float64))

            probs=tf.transpose(a=tf.stack(values=probs,axis=0),perm=[2,0,1,3])
            log_phases=tf.transpose(a=tf.stack(values=log_phases,axis=0),perm=[2,0,1,3])

            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(input_tensor=tf.reduce_sum(input_tensor=tf.math.log(tf.reduce_sum(input_tensor=tf.multiply(probs,one_hot_samples),axis=3)),axis=2),axis=1)
            self.log_phases=tf.reduce_sum(input_tensor=tf.reduce_sum(input_tensor=tf.reduce_sum(input_tensor=tf.multiply(log_phases,one_hot_samples),axis=3),axis=2),axis=1)

            return self.log_probs, self.log_phases

###################################################################################
    def log_amplitude_nosym(self,samples,inputdim):
        log_prob_temp, log_phase_temp = self.log_amplitude_nosymmetry(samples,inputdim)
        return tf.complex(0.5*log_prob_temp,log_phase_temp)

###############Helping_Function#############################

    def log_amplitudes_fromsymmetrygroup(self, list_samples, inputdim, group_character_signs):

        with tf.device('/CPU:0'):
            group_cardinal = len(list_samples)
            numsamples = tf.shape(list_samples[0])[0]
            numgpus = get_numavailable_gpus()

            list_probs = [[] for i in range(numgpus)]
            list_phases = [[] for i in range(numgpus)]

            list_samples = tf.reshape(tf.concat(list_samples, 0), [-1, self.Nx, self.Ny])
            numsamplespergpu = tf.shape(list_samples)[0]//numgpus #We assume that is divisible!


        for i in range(numgpus):
            with tf.device("/GPU:"+str(i)):
                log_prob_temp, log_phase_temp = self.log_amplitude_nosymmetry(list_samples[i*numsamplespergpu:(i+1)*numsamplespergpu],inputdim)
                list_probs[i] = tf.exp(log_prob_temp)
                list_phases[i] = tf.complex(tf.cos(log_phase_temp), tf.sin(log_phase_temp))

        with tf.device('/CPU:0'):
            list_probs = tf.reshape(tf.concat(list_probs, 0), [group_cardinal,numsamples])
            list_phases = tf.reshape(tf.concat(list_phases, 0), [group_cardinal,numsamples])
            signed_phases = [list_phases[i]/group_character_signs[i] for i in range(len(group_character_signs))]

            regularized_phase = tf.complex(regularized_identity(tf.real(sum(signed_phases)), epsilon = 1e-4),tf.imag(sum(signed_phases)))
            return tf.complex(0.5*tf.math.log(tf.reduce_sum(list_probs, axis = 0)/group_cardinal),tf.math.imag(tf.math.log(regularized_phase)))

################ Rot Symmetry ###################################################

    def log_amplitude_rotsym(self,samples,inputdim, group_character):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-1), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-3), [-1,self.Nx, self.Ny]))

            if group_character == "A":
                group_character_signs = [+1, +1, +1, +1]
            if group_character == "B":
                group_character_signs = [+1, -1, +1, -1]

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs)

    def log_amplitude_rotsym_spinparity(self,samples,inputdim, group_character, spinparity_value):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-1), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-3), [-1,self.Nx, self.Ny]))

            list_samples.append(tf.abs(1-samples))
            list_samples.append(tf.abs(1-list_samples[1]))
            list_samples.append(tf.abs(1-list_samples[2]))
            list_samples.append(tf.abs(1-list_samples[3]))

            if group_character == "A":
                group_character_signs = [+1, +1, +1, +1]
            if group_character == "B":
                group_character_signs = [+1, -1, +1, -1]

            group_character_signs_total = np.concatenate((group_character_signs, spinparity_value*group_character_signs), axis = 0)

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs_total)

################ Rotation reduced Symmetry ###################################################
    def log_amplitude_rotreducedsym_spinparity(self,samples,inputdim, group_character, spinparity_value):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-1), [-1,self.Nx, self.Ny]))

            list_samples.append(tf.abs(1-samples))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(tf.abs(1-samples), [-1,self.Nx, self.Ny, 1]),k=-1), [-1,self.Nx, self.Ny]))

            if group_character == "A":
                group_character_signs = [+1, +1]
            if group_character == "B":
                group_character_signs = [+1, -1]

            group_character_signs_total = np.concatenate((group_character_signs, spinparity_value*group_character_signs), axis = 0)

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs_total)

    def log_amplitude_rotreducedsym(self,samples,inputdim, group_character):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-1), [-1,self.Nx, self.Ny]))

            if group_character == "A":
                group_character_signs = [+1, +1]
            if group_character == "B":
                group_character_signs = [+1, -1]

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs)


################# c2v ##################
    def log_amplitude_c2vsym(self,samples,inputdim, group_character):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(samples[:,::-1])
            list_samples.append(samples[:,:,::-1])

            if group_character == "A1":
                group_character_signs = [+1, +1, +1, +1]
            if group_character == "A2":
                group_character_signs = [+1, +1, -1, -1]
            if group_character == "B1":
                group_character_signs = [+1, -1, +1, -1]
            if group_character == "B2":
                group_character_signs = [+1, -1, -1, +1]

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs)

    def log_amplitude_c2v_spinparity(self,samples,inputdim, group_character, spinparity_value):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(samples[:,::-1])
            list_samples.append(samples[:,:,::-1])

            list_samples.append(tf.abs(1-samples))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(tf.abs(1-samples), [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.abs(1-samples[:,::-1]))
            list_samples.append(tf.abs(1-samples[:,:,::-1]))

            if group_character == "A1":
                group_character_signs = [+1, +1, +1, +1]
            if group_character == "A2":
                group_character_signs = [+1, +1, -1, -1]
            if group_character == "B1":
                group_character_signs = [+1, -1, +1, -1]
            if group_character == "B2":
                group_character_signs = [+1, -1, -1, +1]

            group_character_signs_total = np.concatenate((group_character_signs, spinparity_value*group_character_signs), axis = 0)

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs_total)


################# c2 point group #############
    def log_amplitude_c2sym_spinparity(self,samples,inputdim, group_character, spinparity_value):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))

            list_samples.append(tf.abs(1-samples))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(tf.abs(1-samples), [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))

            if group_character == "A":
                group_character_signs = [+1, +1]
            if group_character == "B":
                group_character_signs = [+1, -1]

            group_character_signs_total = np.concatenate((group_character_signs, spinparity_value*group_character_signs), axis = 0)

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs_total)


    def log_amplitude_c2sym(self,samples,inputdim, group_character):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))

            if group_character == "A":
                group_character_signs = [+1, +1]
            if group_character == "B":
                group_character_signs = [+1, -1]

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs)

##################### c2d #########################
    def log_amplitude_c2dsym(self,samples,inputdim, group_character):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.transpose(a=samples, perm = [0,2,1]))
            list_samples.append(tf.transpose(a=list_samples[1], perm = [0,2,1]))

            if group_character == "A1":
                group_character_signs = [+1, +1, +1, +1]
            if group_character == "A2":
                group_character_signs = [+1, +1, -1, -1]
            if group_character == "B1":
                group_character_signs = [+1, -1, +1, -1]
            if group_character == "B2":
                group_character_signs = [+1, -1, -1, +1]

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs)


    def log_amplitude_c2dsym_spinparity(self,samples,inputdim, group_character, spinparity_value):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.transpose(a=samples, perm = [0,2,1]))
            list_samples.append(tf.transpose(a=list_samples[1], perm = [0,2,1]))

            list_samples.append(tf.abs(1-samples))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(tf.abs(1-samples), [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.abs(1-list_samples[2]))
            list_samples.append(tf.abs(1-list_samples[3]))

            if group_character == "A1":
                group_character_signs = [+1, +1, +1, +1]
            if group_character == "A2":
                group_character_signs = [+1, +1, -1, -1]
            if group_character == "B1":
                group_character_signs = [+1, -1, +1, -1]
            if group_character == "B2":
                group_character_signs = [+1, -1, -1, +1]

            group_character_signs_total = np.concatenate((group_character_signs, spinparity_value*group_character_signs), axis = 0)

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs_total)


#C4v##################

    def log_amplitude_c4vsym(self,samples,inputdim, group_character):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-1), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-3), [-1,self.Nx, self.Ny]))
            list_samples.append(samples[:,::-1])
            list_samples.append(samples[:,:,::-1])
            list_samples.append(tf.transpose(a=samples, perm = [0,2,1]))
            list_samples.append(tf.transpose(a=list_samples[2], perm = [0,2,1]))

            if group_character == "A1":
                group_character_signs = [+1, +1, +1, +1, +1, +1, +1, +1]
            if group_character == "A2":
                group_character_signs = [+1, +1, +1, +1, -1, -1, -1, -1]
            if group_character == "B1":
                group_character_signs = [+1, -1, +1, -1, +1, +1, -1, -1]
            if group_character == "B2":
                group_character_signs = [+1, -1, +1, -1, -1, -1, +1, +1]

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs)

#### with spin parity projection
    def log_amplitude_c4vsym_spinparity(self,samples,inputdim, group_character, spinparity_value):

        with tf.device('/CPU:0'):
            list_samples = [samples]
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-1), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(samples, [-1,self.Nx, self.Ny, 1]),k=-3), [-1,self.Nx, self.Ny]))
            list_samples.append(samples[:,::-1])
            list_samples.append(samples[:,:,::-1])
            list_samples.append(tf.transpose(a=samples, perm = [0,2,1]))
            list_samples.append(tf.transpose(a=list_samples[2], perm = [0,2,1]))

            list_samples.append(tf.abs(1-samples))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(tf.abs(1-samples), [-1,self.Nx, self.Ny, 1]),k=-1), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(tf.abs(1-samples), [-1,self.Nx, self.Ny, 1]),k=-2), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.reshape(tf.image.rot90(tf.reshape(tf.abs(1-samples), [-1,self.Nx, self.Ny, 1]),k=-3), [-1,self.Nx, self.Ny]))
            list_samples.append(tf.abs(1-samples)[:,::-1])
            list_samples.append(tf.abs(1-samples)[:,:,::-1])
            list_samples.append(tf.transpose(a=tf.abs(1-samples), perm = [0,2,1]))
            list_samples.append(tf.transpose(a=list_samples[2+8], perm = [0,2,1]))


            if group_character == "A1":
                group_character_signs = np.array([+1, +1, +1, +1, +1, +1, +1, +1])
            if group_character == "A2":
                group_character_signs = np.array([+1, +1, +1, +1, -1, -1, -1, -1])
            if group_character == "B1":
                group_character_signs = np.array([+1, -1, +1, -1, +1, +1, -1, -1])
            if group_character == "B2":
                group_character_signs = np.array([+1, -1, +1, -1, -1, -1, +1, +1])

            group_character_signs_total = np.concatenate((group_character_signs, spinparity_value*group_character_signs), axis = 0)

        return self.log_amplitudes_fromsymmetrygroup(list_samples, inputdim, group_character_signs_total)
