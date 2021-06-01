import tensorflow as tf
import tensorflow.keras as keras

class ND_Layer(keras.layers.Layer):
    """"
    The base layers class
    for spatial flow layers.

    spatial flow layers  are represented in
    much the same way as keras layers, but with some key
    differences. The primary difference is that
    each and every layer exists as an entity capable
    of simaltanously evaluating ALL spatial dimensions.

    """
    def __init__(self,
                 sharing = False,
                 trainable = True,
                 name = "Layer"
                 ):
        super().__init__(trainable, name)



class Dense(ND_Layer):
    """

    The ND implementation of a dense layer.

    This uses tensordot and broadcasting to implicitly
    evaluate the entirety of a multidimensional tensor in
    a single massive tensordot operation.

    """

    def __init__(self,
                 total_batch_dims : int,
                 units,
                 reduction_dims = "all",
                 sharing=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer = "glorot_uniform",
                 bias_initializer = None,
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 activitY_regularizer = None,
                 kernel_constraint = None,
                 bias_constraint =None,
                 **kwargs):
        """


        :param units: What shape the tensor output should be
        :param sharing: A 1D bool list, or None. If not None, must match length of call input, and defines which
            dimensions to share parameters on; True means share, False means don't. Defaults to false.
        :param activation: Like keras Dense
        :param use_bias: Like keras Dense
        :param kernel_initializer: Like keras Dense
        :param bias_initializer: Like keras Dense
        :param kernel_regularizer: Like keras Dense
        :param bias_regularizer: Like keras Dense
        :param activitY_regularizer: Like keras Dense
        :param kernel_constraint: Like keras Dense
        :param bias_constraint: Like keras Dense
        :param kwargs:
        """
        #Initialize
        super().__init__(kwargs)

        #Begin storing items
        self._units = tf.TensorShape(units)
        self._reduction_dims = tf.constant(reduction_dims)
        self._sharing = tf.constant(sharing)

        #store keras functions


        self._use_bias = use_bias
        self._activation = keras.activations.get(activation)
        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._bias_initializer = keras.initializers.get(bias_initializer)
        self._kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = keras.regularizers.get(bias_regularizer)
        self._activity_regularizer = keras.regularizers.get(activitY_regularizer)
        self._kernel_constraint = keras.constraints.get(kernel_constraint)
        self._bias_constraint = keras.constraints.get(bias_constraint)

    def build(self, input_shape):
        #perform basic verification



        if self._reduction_dims is "all":
            self._reduction_dims = tf.fill([input_shape.rank], True)


        self._output_shape = tf.TensorShape(tf.where(self._units is None, input_shape, self._units))


        ##Go ahead and build the variable arrays at the heart of this

        kernel_core = tf.where(self._sharing, 1, input_shape)
        kernel_core = tf.where(self._reduction_dims, kernel_core, 1)
        kernel_shape = tf.TensorShape(kernel_core).concatenate(tf.TensorShape(self._units))
        self._kernel = self.add_weight(name = "kernel", shape = kernel_shape, initializer=self._kernel_initializer,
                        regularizer=self._kernel_regularizer, constraint=self._kernel_constraint)
        if self._use_bias:
            bias_core = tf.boolean_mask(input_shape, tf.logical_not(self._reduction_dims))
            bias_shape = tf.TensorShape(bias_core).concatenate(tf.TensorShape(self._units))
            self._bias = self.add_weight(name="bias", shape=bias_shape, initializer=self._bias_initializer,
                                         regularizer=self._bias_regularizer, constraint=self._bias_constraint)
    @tf.function
    def tensordot(self, input):
        input_broadcast = tf.broadcast_to(input, self._kernel.shape)
        kernel_broadcast = tf.broadcast_to(self._kernel, input_broadcast.shape)
        return tf.tensordot(input_broadcast, kernel_broadcast, axes=[])
    def call(self, input):


