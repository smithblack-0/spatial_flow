import tensorflow as tf
import tensorflow.keras as keras
from spatial_flow.utils.error_utils import Reducer_Error
from spatial_flow.selectors import Selector
"""

A reducer is an object which accepts a 
tensor in comparison format and reduces
that tensor back down to a spatial_grid
shape. It is where the majority
of the actual gradient learning is allowed to take
place 

"""

spatial_register = keras.utils.register_keras_serializable("spatial_flow/reduction")

@spatial_register
class Reducer(keras.layers.Layer):
    """ 
    
    The base class for reducers, Reducer possesses the important
    method designed to be overridden called "reduce"

    reducer should be initialized with a single
    selector, after which it will forever be tied to it.


    """
    @property
    def batch_dims(self):
        return self._batch_dims
    @property
    def channel_dims(self):
        return self._channel_dims
    @property
    def selector(self):
        return self._selector
    def __init__(self, selector, reduce_dims = "all", name="reducer"):
        """

        The initializer for the reducer class.

        :param selector: A valid selector
        :param reduce_dims: The dimensions to be reduced. Supports "All" or a 1D bool list
        :param name: The name of the layer.
        """

        #verify inputs are sane
        if not isinstance(selector, Selector):
            return Reducer_Error("Input 'selector' was not a selector")


        #Initialize and store
        super().__init__(name=name, **kwargs)



        self._selector = selector


        #Set default values

        self._batch_dims = None
        self._channel_dims = None

        #Go wrap call to strip batch and channel dims off


        self.__stored_call = self.__call__
        def stripper(self, input):

            self._total_batch_dims = input.batch_dims
            self._channel_dims = input.channel_dims
            self.__call__ = self.__stored_call
            self.__call__(input)
        self.__call__ = stripper

@spatial_register
class dense_reducer(Reducer):
    """

    The dense reducer performs the reduction along the targetted dimensions in one
    massive tensordot. It supports parameter sharing as in convolutions per dimension,
    and can be loaded with many of the keras goodies such as kernel and bias initializers.

    """

    def __init__(self,
                 selector,
                 reduction_dims="all",
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


        :param reduction_dims: a list of bools, matching the remaining comparison space, or "all" to reduce everything
            parameter defines which dimensions should be reduced, and which should not.
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
        super().__init__(selector, kwargs)

        #Begin storing dimensions items.

        if reduction_dims is not None:
            self._reduction_dims = tf.constant(reduction_dims, dtype=tf.dtypes.bool)
        else:
            self._reduction_dims = None

        if sharing is not None:
            self._sharing = tf.constant(sharing, dtype = tf.dtypes.bool)

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

        #perform verification and extrapolation

        tf.debugging.assert_rank_at_least(input_shape, self.batch_dims.rank)



        #construct kernel

        kernel_shape = tf.where(self._sharing, )


@spatial_register
class keras_reducer(Reducer):
    """

    A reducer to enable the direct usage of keras
    layers in a reducing context.

    This layer comes standard with the abstract
    method "reduce" which should return a keras
    layer according to specifications. This is
    then utilized to reduce the indicated dimensions.

    """
    def __fetch_reduce(self):

        #Fetch from user code, see if it runs at all.
        try:
            reduce_func = self.reduce()
        except Exception as err:
            msg = "Error in user code, reduce: did not successfully return: %s" % err
            raise Reducer_Error(msg) from err
        if not callable(reduce_func):
            raise Reducer_Error("Error in user code: Did not return callable function or class")


        #Build wrapper around function for sanity checking purposes
        def reduce_wrapper(self, input):
            #Does it even execute?
            try:
                output = reduce_func(input)
            except Exception as err:
                msg = "Error in user layer, layer did not run on input: %s" % err
                raise Reducer_Error(msg) from err
            #It executed. IS the result of sane shape?


            shape = self.batch_dims.concatenate(self.channel_dims).concatenate(tf.TensorShape([1]))
            tf.debugging.assert_shapes([(output, shape)],
                                       message="Error in user layer. Expected shape %s, got %s" %(shape, output.shape))

            #looks good. Return
            return output


        #Return wrapped reduce

        return reduce_wrapper
    def reduce(self, batch_dims, channel_dims, comparison_shape):
        """

        Reduce is the location in which the template
        for the function which will reduce the given dimensions
        should be placed.

        Please note that the layer MUST have a working
        .get_config and .from_config to work. Additionally,
        no batch numbers are provided, at least for now.

        The total shape of the incoming block will be
        [batch dims, channel_dims, comparison_dims]. Outgoing it should
        be [batch_dims, channel_dims, 1]


        The default mode simply flattens the input and
        returns it with a dense. It is recommended you do NOT activate
        the keras layers here, to allow integration tricks to work.

        :param batch_dims: The shape of the batch dimensions.
        :param comparison_shape: The shape of the comparison sector. Use it however
            desired
        :param channel_dims: The shape of any incoming channels.
        :return: A keras layer representing the reductions to be performed for a single neuron. This
            layer when called, in turn, must produce a single output per batch and channel
        """


        #Define layer

        class default(tf.keras.layers.Layer):
            def __init__(self, batch_dims, channel_dims, comparison_shape):
                super().__init__(name = "default")
                reshape_dims =  batch_dims.concatenate(channel_dims).concatenate(tf.TensorShape(tf.reduce_prod(comparison_shape)))
                self._flatten = lambda input : tf.reshape(input, reshape_dims)
                self._dense = tf.keras.layers.Dense(1)
            def call(self, input):
                output = self._flatten(input)
                output = self._dense(output)
                return output

        #Return layer

        return default(batch_dims, channel_dims, comparison_shape)

