import tensorflow as tf
import tensorflow.keras as keras
import spatial_flow.core as core

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
                 units,
                 reduction_dims,
                 sharing,
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
        :param sharing: A 1D bool list the length of reduction dimensions,
         and defines which dimensions to share parameters on;True means share, False means don't.
        :param reduction_dims: A 1D bool list the length of the expected input rank.
                    Which dimensions of the input will actually be reduced. True means reduce. False ignore.
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

        ##Go ahead and build the kernel and bias cores

        data_start = input_shape.rank - self._reduction_dims.shape.rank
        kernel_core = []
        bias_core = []
        sharing_stack = tf.unstack(self._sharing, axis=0)
        for index in range(self._reduction_dims.shape.rank):
            if self._reduction_dims[index] and not sharing_stack.pop(0):
                kernel_core.append(input_shape[index + data_start])
            else:
                kernel_core.append(1)
                bias_core.append(input_shape[index])
        kernel_core = tf.stack(kernel_core, axis=0)
        bias_core = tf.stack(bias_core, axis=0)
        kernel_shape = tf.TensorShape(kernel_core).concatenate(tf.TensorShape(self._units))
        bias_shape = tf.TensorShape(bias_core).concatenate(tf.TensorShape(self._units))

        #Add extra dimensions as needed

        while input_shape.rank > kernel_shape.rank:
            kernel_shape = kernel_shape.concatenate([1])


        #build variables

        self._kernel = self.add_weight(name = "kernel", shape = kernel_shape, initializer=self._kernel_initializer,
                        regularizer=self._kernel_regularizer, constraint=self._kernel_constraint)
        if self._use_bias:
            self._bias = self.add_weight(name="bias", shape=bias_shape, initializer=self._bias_initializer,
                                         regularizer=self._bias_regularizer, constraint=self._bias_constraint)
    def tensordot(self, input):

        #add dimensions as required

        while input.shape.rank < self._kernel.shape.rank:
            input = tf.expand_dims(input, axis=0)

        #identify indices

        input_indices = tf.cast(tf.where(self._reduction_dims), tf.dtypes.int32)
        kernel_indices = tf.cast(tf.range(0, self._sharing.shape.rank), tf.dtypes.int32)
        #take tensordot

        return tf.tensordot(input, self._kernel, axes=[input_indices, kernel_indices])
    def call(self, input):

        multiply = self.tensordot(input)
        return tf.add(multiply,self._bias)


