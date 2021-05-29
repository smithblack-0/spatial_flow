import tensorflow as tf
import tensorflow.keras as keras

import spatial_flow.config
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

    The selector class is responsible for turning a standard tensor into
    a comparison tensor when called. It is initialized with a reference,
    which always represents the initial result which will be provided
    if called, and is subsequently responsible for managing and modifying it's
    references as needed to perform any dynamic behavior which is required of the model,
    such as modifying the network structure for different problems or performing mutations
    in genetic algorithms.

    The selector is initialized with a valid config, a reference, and the optional but
    important variables "dynamics" and "unpacking."  These have important consequences
    for the modify method and it's functionality. 

    Calling it with a standard block will return a comparison block.

    parameters
        reference: a tf.Variable containing the underlying reference. Provides manual control.

    methods:
        select(standard, reference): Performs an actual selection using references.
        modify(*args, *kwargs): Used to easily modify the underlying references

    initialization:
        config: A valid config
        reference: A valid reference:
        dynamics:
            A bool or list of bools specifying what spatial indices of the reference
            are allowed to be modified. May be "True" indicating all can be modified, 
            "False" indicating everything is static, or a list of bool of length
            spatial_rank, indicating for each dimension whether it can be modified or not.
        unpacking: 
            Selects the mode modify will be called in. May be "all", "partial" or "none".


    This is meant to be subclassed.
    """

    # define properties

    @property
    def reference(self):
        return self._reference

    @reference.setter
    def reference(self, value):
        validity = self._config.reference_config.is_reference(value)
        if validity != True:
            raise Selection_Error("Set reference: %s" % validity)
        self._reference.assign(value)

    @property
    def dynamics(self):
        return self._dynamics

    @dynamics.setter
    def dynamics(self, value):

        # Sanity check
        tf.debugging.assert_type(value, tf.dtypes.bool, "Selection Error: Dynamics was not bool or list of bools.")
        tf.debugging.assert_rank_in(value, [0, 1], "Selection Error: Dynamics was not 0D or 1D tensor")
        # Broadcast where needed
        if tf.constant(value).shape.rank == 0:
            value = tf.repeat([value], self._config.compare_config.spatial_rank)
        # Set
        self._dynamics = value

    def __init__(self, config, reference, dynamics=False, unpacking="all", name='selector'):

        super().__init__(name=name)

        # Sanity check

        if not isinstance(config, spatial_flow.config.spatial_config):
            raise Selection_Error("Input 'config' was not valid spatial config")

        check_reference = config.reference_config.is_reference(reference)
        if not check_reference:
            raise Selection_Error("Input 'reference' was not valid: %s" % check_reference)
        if unpacking not in ("all", "partial", "none"):
            raise Selection_Error("Input 'unpacking' was not 'all', 'partial', or 'none'")

        # Set things

        self._config = config
        self._reference = tf.Variable(reference)
        self.dynamics = dynamics
        self._unpacking = unpacking

        # Adjust modify to possess robust error handling.



    def run_modify(self, *args, **kwargs):
        """

        Run modify unpacks the reference dimensions to the level specified by
        "unpacking",  then provides each dynamic result into the code in "modify."

        An unpacking level of "all" results in an reference being passed in each time
        An unpacking level of "partial" results in comparison reference blocks being passed in each time
        An unpacking level of "none" results in the entire spatial reference block being passed in each time.

        The return, which is expected to be of the same shape and error checked for such,
        is then stacked back together and assigned to the reference slot.


        :param args: Any arguments modify needs
        :param kwargs: Any keyword arguments modify needs
        :return: None
        """

        #Define modify wrapper. This is responsible for error checking and handling for modify
        def wrapper(bool_mask, unpacked_references, *args, **kwargs):

            try:
                #restrict your references to those which can be modified
                restricted_references = tf.boolean_mask(unpacked_references, bool_mask, axis=-1)
                #pass in and eval
                prelimary_output = self.modify(restricted_references, args, kwargs)
                #test

                #restore shape and return
                stitch_indices = [tf.where(tf.equals(bool_mask, False)), tf.where(tf.equals(bool_mask, True))]
                output = tf.dynamic_stitch(stitch_indices, [unpacked_references, prelimary_output], axis = -1)
                return output
            except Exception as err:
                raise Selection_Error("Problem in Modify: %s" % err)

        #Run modify at the appropriate level.
        run = lambda reference : wrapper(reference, args, kwargs)
        if(self._unpacking == "all"):
            result = core.unpack_reference(self._config, self.reference, run, level = "individual")
        elif(self._unpacking == "partial"):
            result = core.unpack_reference(self._config, self.reference, run, level = "comparison")
        else:
            result = core.unpack_reference(self._config, self.reference, run, level = "spatial")
        #

    def modify(self, unpacked_references, *args, **kwargs):
        """

        Modify is designed to work with parameter "unpacking" and method "run_modify" to allow easy
        modification

        Only references which are listed as dynamic may be passed in for modification.

        An unpacking level of "all" results in individual references being passed in each time
        An unpacking level of "partial" results in  comparison reference blocks being passed in each time
        An unpacking level of "none" results in the entire spatial reference block being passed in each time.

        Modify may perform any modifications on the passed in references it wishes, but must return the same
        tensor shape and type as the "unpacked_references" input.



        :param unpacked_references: Automatically provided when run_modify is called.
            The reference for the given unpacking level
        :param args: The args passed into run_modify
        :param kwargs: the kwargs passed into run_modify
        :return: a series of references
        """

    def select(self, standard, reference):
        # Perform a selection.

        reference = self.reference
        # Do this by verifying sane inputs, then unpacking, then selecting, then repacking.

        reference_validation = self._config.reference_config.is_reference(reference)
        standard_validation = self._config.standard_config.is_standard(standard)

        if (reference_validation != True):
            raise ValueError("Select Validation Failed: Reference input: %s" % reference_validation)
        if (standard_validation != True):
            raise ValueError("Select Validation Failed: Standard input: %s" % standard_validation)
        # move tail to front head, define restore, and define fold parameters

        # define unpacking function chain
        def gather(spatial, reference):
            return tf.gather_nd(spatial, reference)
        def reference_unpack(config, spatial, references):
            funct = lambda reference : gather(spatial, reference)
            shape = tf.TensorShape(config.reference_config.comparison_shape)
            result = core.unpack_reference(config, references, funct, shape=shape, level="comparison")
            return result
        def spatial_unpack(config, standard, references):
            funct = lambda spatial : reference_unpack(config, spatial, references)
            shape = tf.TensorShape(config.compare_config.comparison_shape)
            result = core.unpack_standard(config, standard, funct, shape=shape, level="spatial")
            return result

        # execute function and return result

        output = spatial_unpack(self._config, standard, reference)
        return output

@spatial_register
class Static(Selector):
    """ 

    The static selection layer.

    When initialized, the static selection layer goes and 
    compiles a graph with the provided references to make
    a function which will transform a standard_block into 
    a comparison_block

    It requires only one variable upon initialization:
    the reference block to statically represent.

    """

    def __init__(self, config, reference, name='static_selector'):
        super().__init__(config, name=name)
        self._reference = reference
        callable = lambda block: self.select(block, reference)
        self._func = tf.function(callable)

    def call(self, standard_block):
        return self._func(standard_block)

    def get_config(self):
        return {"reference": self._reference}


@spatial_register
class Dynamic(Selector):
    """ 

    A mutable selector layer.

    The dynamic layer lies at the heart of any 
    advanced selector. It allows for the initialization 
    of 

    If you want low level, absolute control of 
    each reference with no automatation and will
    accept no substitute, the dynamic layer is 
    your friend.

    Dynamic accepts a initial reference 
    configuration and sets it to the 
    variable property "references." From this point forward,
    a new series of references can be assigned to this
    at any moment that is the same shape by simple
    standard assignment methods.

    Initialization Argument:

        - References: A reference block. All references after this point must match this shape.

    Properties:
        - References: A variable where references can be assigned. Must match original reference shape.

    """

    def __init__(self, config, reference, name='dynamic_selector'):
        super().__init__(config, name=name)
        self.reference = tf.Variable(reference)

    @tf.function
    def call(self, standard_block):
        return self.select(standard_block, self.reference)

    def get_config(self):
        return {"reference": self._reference}


class genetic_basic(keras.layers.Layer):
    """ genetic selector. 
    
    A genetic selector provides support for changable 
    selections by way of variables and mutate functions. 

    """

    pass
