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


class Reducer(keras.layers.Layer):
    """ 
    
    The base class for reducers, Reducer possesses the important
    method designed to be overridden called "reduce"

    reducer should be initialized with a single
    selector, and which it will be forever after
    be tied to.

    As in most cases, Reducer comes with a method to be overridden.
    In it's case, it is the appropriately named "reduce"
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
    def __init__(self, selector, reduce_dims = "all" , sharing=False, name="reducer"):
        """
        The initialization method.


        :param selector: the selector which will be reduced
        :param sharing: A tensor telling the sharing behavior
            Parameter sharing is quite viable
            in a reducer environment, and a powerful technique
            as well.

            For this implimentation, it simply means the keras
            layers will be reused along the specified axis.

            As such, one can provide True to enable parameter
            sharing along all dimensions, or a 1D bool tensor
            of the appropriate length to enable parameter sharing
            along each dimension.

        """

        #verify inputs are sane
        super().__init__(name=name)

        if not isinstance(selector, Selector):
            raise Reducer_Error("Init: Expected selector, got type %s" % type(selector))
        tf.debugging.assert_type(sharing, tf.dtypes.bool, "Reducer error: sharing not bool")
        tf.debugging.assert_rank_in(sharing, (0,1), "Reducer error: sharing: ")

        sharing = tf.constant(sharing)
        if sharing.shape.rank == 0:
            sharing = tf.fill([selector.comparison_shape.rank], sharing)
        else:
            tf.debugging.assert_shapes([(sharing, [selector.comparison_shape.rank])])
        #save values

        self._selector = selector


        #Set default values

        self._batch_dims = None
        self._channel_dims = None

        #Go wrap call to strip


    def call(self, input):
        #if not yet set, go set batch and channel dims

        if self._batch_dims is None:
            self._batch_dims = input.batch_dims
        if self._channel_dims is None:
            self._channel_dims = input.channel_dims



spatial_register = keras.utils.register_keras_serializable("spatial_flow/reduction")


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


    def __init__(self):