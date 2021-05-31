import tensorflow as tf
import tensorflow.keras as keras

import spatial_flow.config
from spatial_flow.reference import Reference
from spatial_flow.utils.error_utils import Selection_Error
import spatial_flow.core as core

"""


Selectors are, in essence, the manager for a pile of references. A selector accepts upon initialiation a 
reference block, and uses this to somehow initialize itself. From this point forward, it is responsible for using
it's internal state to fetch from an input tensor the appropriate spatial states and compile 
a tensor in comparison format.

Notably, the selections made need not be static each time. Due to operating using references under the hood, the 
only requirement tensorflow enforces to avoid rebuilding the graph is that the number of references in each slot 
be the same each time. Beyond this, graphs should compile as normal, and are.

Selectors are built using the base "Selector" class, which comes with the "select" method for taking a 
reference and a datablock and using it to build a compare tensor. 

Since Selectors need to persistently maintain their state, they are a variety of keras layer. However, by no means
are they standard; indeed, they represent something completely different from a neural layer.

"""

spatial_register = keras.utils.register_keras_serializable("spatial_flow/selectors")


@spatial_register
class Selector(keras.layers.Layer):
    """

    The base selector class

    The selector class is responsible for turning a reference and standard tensor into a comparison
    tensor, while simultaneously managing and updating any reference entries as needed o perform any dynamic
    behavior which is required of the model, uch as modifying the network architecture between batches or
    or performing mutations for genetic algorithms.

    The selector is initialized with a reference, and is subsequently responsible for performing any
    desired modification to the reference if so needed. Only reference items which are marked mutable
    can be changed from a selector.

    :method modify:
        'modify" lies at the core of how the selector works. It should be overwridden and logic placed
        within it according to the directives.
    :method run_modify:
        run_modify runs the code in modify. It can be called externally.

    This is meant to be subclassed.
    """
    @property
    def spatial_shape(self):
        return self._spatial_shape
    @property
    def comparison_shape(self):
        return self._comparison_shape
    @property
    def index_shape(self):
        return self._index_shape
    def __init__(self, reference, name="selector", mode="simple"):
        """

        The initializer

        Mode is the only item deserving special attention. Mode controls whether
        modify is expected to work with only the comparison references, or whether
        it requires the comparison references and the spatial indices.

        For the vast majority of purposes, simple is sufficinet.

        :param reference: a valid reference
        :param name: The name of this object
        :param mode: either "simple" or "advanced"
        """
        super().__init__(name=name)

        # Quick Sanity check

        if not isinstance(reference, Reference):
            raise Selection_Error("init - Not provided with a reference of type 'Reference'")
        if type(mode) != str:
            raise Selection_Error("init - mode was not string")
        if mode not in ("simple", "advanced"):
            raise Selection_Error("init - mode was not 'simple' or 'advanced")

        #Store reference

        self.reference = reference
        self._mode = mode
        self._spatial_shape = reference.spatial_shape
        self._comparison_shape = reference.comparison_shape
        self._index_shape = reference.index_shape
        #hack save to work transparently

            #impliment

    def modify(self, comparison_references, *args, **kwargs):
        """

        This should be overridden

        Defines modify under the simple case, where it is true that
        only the unpacked references are needed. Most genetic
        algorithms mutations should be easily accomplished like this.

        Will only be used if mode is simple

        use method "run_modify" to use it

        :param comparison_references: A comparison reference
        :param args: Any args your code will need
        :param kwargs: Any kwargs your code will need.
        :return: A comparison block.
        """



    def modify(self, comparison_reference, spatial_index, *args, **kwargs):
        """"
        This should be overridden

        Defines modify for the advanced case, where different changes need to
        be performed at different spatial indices. Keep in mind that in general,
        you should restrict mutability when making references instead; only use this
        if there is no other option.

        use method "run_modify" to use it

        :param comparison_reference: A given comparison reference
        :param spatial_index: A given spatial index
        :param args: Any args your code may need
        "param kwargs: Any Kwargs your code may need

        """


    def run_modify(self, *args, **kwargs):
        """

        Runs your declared modify function with the given args and kwargs.


        :param args: Any arguments modify needs
        :param kwargs: Any keyword arguments modify needs
        :return: None
        """

        if(self._mode == "simple"):
            update_func = lambda unpacked, spatial_index : self.modify(unpacked, args, kwargs)
            self.reference.update(update_func)
        else:
            update_func = lambda unpacked, spatial_index : self.modify(unpacked, spatial_index, args, kwargs)
            self.reference.update(update_func)

    def call(self, spatialgrid):
        """

        Perform the extraction and produce a tensor in comparison format.

        Compiles with tensorflow.


        :param spatialgrid: A tensor in spatialgrid format
        :return: A tensor in comparison format
        """
        


        def callback(comparison_grid):
            gather = tf.gather_nd(spatialgrid, comparison_grid)
            return gather

        shape = self.reference.comparison_shape
        output = self.reference.unpack(callback, shape)
        return output


