import tensorflow as tf

def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class BasicRNNCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_rnn_cell", reuse=self._reuse):
            # todo: implement the new_state calculation given inputs and state
            new_state = self._activation(tf.layers.dense(inputs, self._num_units) + tf.layers.dense(state, self._num_units, use_bias=False))
            return new_state, new_state

class GRUCell(tf.contrib.rnn.RNNCell):
    '''Gated Recurrent Unit cell (http://arxiv.org/abs/1406.1078).'''

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "gru_cell", reuse=self._reuse):
            # We start with bias of 1.0 to not reset and not update.
            # todo: implement the new_h calculation given inputs and state
            update_gate = tf.sigmoid(tf.layers.dense(inputs, self._num_units, bias_initializer=tf.ones_initializer()) + tf.layers.dense(state, self._num_units, use_bias=False))
            reset_gate = tf.sigmoid(tf.layers.dense(inputs, self._num_units, bias_initializer=tf.ones_initializer()) + tf.layers.dense(state, self._num_units, use_bias=False))
            update_h = self._activation(tf.layers.dense(inputs, self._num_units, use_bias=False) + tf.layers.dense(reset_gate * state, self._num_units, use_bias=False))
            new_h = state * (1 - update_gate) + update_h * update_gate

            return new_h, new_h

class BasicLSTMCell(tf.contrib.rnn.RNNCell):
    '''Basic LSTM cell (http://arxiv.org/abs/1409.2329).'''

    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_lstm_cell", reuse=self._reuse):
            c, h = state
            #For forget_gate, we add forget_bias of 1.0 to not forget in order to reduce the scale of forgetting in the beginning of the training.
            #todo: implement the new_c, new_h calculation given inputs and state (c, h)
            input_gate = tf.sigmoid(tf.layers.dense(inputs, self._num_units) + tf.layers.dense(h, self._num_units, use_bias=False))
            output_gate = tf.sigmoid(tf.layers.dense(inputs, self._num_units) + tf.layers.dense(h, self._num_units, use_bias=False))
            forget_gate = tf.sigmoid(tf.layers.dense(inputs, self._num_units, bias_initializer=tf.ones_initializer()) + tf.layers.dense(h, self._num_units))
            update_c = self._activation(tf.layers.dense(inputs, self._num_units) + tf.layers.dense(h, self._num_units, use_bias=False))
            new_c = forget_gate * c + input_gate * update_c
            new_h = output_gate * self._activation(new_c    )
            return new_h, (new_c, new_h)
