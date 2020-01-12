import numpy as np
import tensorflow as tf

###################################################################################################333

class MDRNNcell(tf.contrib.rnn.RNNCell):
    """An implementation of the most basic 2DRNN Vanilla RNN cell.
    Args:
        num_units (int): The number of units in the RNN cell, hidden layer size.
        num_in: Input vector size, input layer size.
    """

    def __init__(self, num_units = None, num_in = None, name=None, dtype = None, reuse=None):
        super(MDRNNcell, self).__init__(_reuse=reuse, name=name)

        self._num_in = num_in
        self._num_units = num_units
        self._state_size = num_units
        self._output_size = num_units

        self.Wh = tf.get_variable("Wh_"+name, shape=[num_units, num_units],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype = dtype)

        self.Uh = tf.get_variable("Uh_"+name, shape=[num_in,num_units],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype = dtype)

        self.Wv = tf.get_variable("Wv_"+name, shape=[num_units, num_units],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype = dtype)

        self.Uv = tf.get_variable("Uv_"+name, shape=[num_in,num_units],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype = dtype)


        self.b = tf.get_variable("b_"+name, shape=[num_units],
                                    initializer=tf.contrib.layers.xavier_initializer(), dtype = dtype)


    # needed properties
    @property
    def input_size(self):
        return self._num_in # real

    @property
    def state_size(self):
        return self._state_size # real

    @property
    def output_size(self):
        return self._output_size # real

    def call(self, inputs, states):

        # prepare input linear combination
        input_mul_left = tf.matmul(inputs[0], self.Uh) # [batch_sz, num_units] #Horizontal
        input_mul_up = tf.matmul(inputs[1], self.Uv) # [batch_sz, num_units] #Vectical

        state_mul_left = tf.matmul(states[0], self.Wh)  # [batch_sz, num_units] #Horizontal
        state_mul_up = tf.matmul(states[1], self.Wv) # [batch_sz, num_units] #Vectical

        preact = input_mul_left + state_mul_left + input_mul_up + state_mul_up  + self.b #Calculating the preactivation

        output = tf.nn.elu(preact) # [batch_sz, num_units]

        new_state = output

        return output, new_state
