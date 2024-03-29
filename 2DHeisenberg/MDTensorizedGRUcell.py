import numpy as np
import tensorflow as tf
from Tensordot2 import tensordot


###################################################################################################333
class MDRNNcell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """
    An implementation of a 2D tensorized GRU RNN cell
    """

    def __init__(self, num_units = None, num_in = None, name=None, dtype = None, reuse=True):
        super(MDRNNcell, self).__init__(_reuse=reuse, name=name)

        self._num_in = num_in
        self._num_units = num_units
        self._state_size = num_units
        self._output_size = num_units

        self.W = tf.compat.v1.get_variable("W_"+name, shape=[num_units, 2*num_units, 2*num_in],
                                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype = dtype)

        self.b = tf.compat.v1.get_variable("b_"+name, shape=[num_units],
                                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype = dtype)

        self.Wg = tf.compat.v1.get_variable("Wg_"+name, shape=[num_units, 2*num_units, 2*num_in],
                                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype = dtype)

        self.bg = tf.compat.v1.get_variable("bg_"+name, shape=[num_units],
                                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype = dtype)

        self.Wmerge = tf.compat.v1.get_variable("Wmerge_"+name, shape=[2*num_units, num_units],
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

        inputstate_mul = tf.einsum('ij,ik->ijk', tf.concat((states[0], states[1]), 1),tf.concat((inputs[0], inputs[1]),1))
        # prepare input linear combination
        state_mul = tensordot(tf,inputstate_mul, self.W, axes=[[1,2],[1,2]]) # [batch_sz, num_units]
        state_mulg = tensordot(tf,inputstate_mul, self.Wg, axes=[[1,2],[1,2]]) # [batch_sz, num_units]

        u = tf.nn.sigmoid(state_mulg + self.bg)
        state_tilda = tf.nn.tanh(state_mul + self.b) # [batch_sz, num_units] C

        new_state = u*state_tilda + (1.-u)*tf.matmul(tf.concat(states, 1), self.Wmerge)
        output = new_state
        return output, new_state
