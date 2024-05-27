import numpy as np
import tensorflow as tf


###################################################################################################333
class MDRNNcell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """
    An implementation of a 2D periodic GRU RNN cell
    """

    def __init__(self, num_units = None, num_in = None, name=None, dtype = None, reuse=True):
        super(MDRNNcell, self).__init__(_reuse=reuse, name=name)

        self._num_in = num_in
        self._num_units = num_units
        self._state_size = num_units
        self._output_size = num_units

        self.W = tf.compat.v1.get_variable("W_"+name, shape=[4*(num_units+3*num_in), num_units],
                                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype = dtype)

        self.b = tf.compat.v1.get_variable("b_"+name, shape=[num_units],
                                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype = dtype)

        self.Wg = tf.compat.v1.get_variable("Wg_"+name, shape=[4*(num_units+3*num_in), num_units],
                                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype = dtype)

        self.bg = tf.compat.v1.get_variable("bg_"+name, shape=[num_units],
                                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype = dtype)

        self.Wmerge = tf.compat.v1.get_variable("Wmerge_"+name, shape=[4*num_units, num_units],
                                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype = dtype)

    # needed properties

    @property
    def input_size(self):
        return self._num_in # real

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, states):

        state_mul = tf.matmul(tf.concat([inputs[0], inputs[1], inputs[2], inputs[3], states[0], states[1], states[2], states[3]],1), self.W) # [batch_sz, num_units]

        state_mulg = tf.matmul(tf.concat([inputs[0], inputs[1], inputs[2], inputs[3], states[0], states[1], states[2], states[3]],1), self.Wg) # [batch_sz, num_units]

        state_tilda = tf.nn.tanh(state_mul + self.b) # [batch_sz, num_units] C
        u = tf.nn.sigmoid(state_mulg + self.bg)

        new_state = u*state_tilda + (1.-u)*tf.matmul(tf.concat(states, 1), self.Wmerge)
        output = new_state
        return output, new_state
