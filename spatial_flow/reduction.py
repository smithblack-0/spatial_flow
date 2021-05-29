import tensorflow as tf
import tensorflow.keras as keras
import tensorflowND.core as core
import tensorflowND.utils.error_utils as error
"""

This is the 
A neurons object is a object which accepts a configuration and a comparison tensor and,
by one means or another, reduces the comparison block back down to a standard block by
performing some sort of gradient driven sum along each comparison dimension.

"""


class Reducer(keras.layers.Layer):
    """ 
    
    The base layer for neurons



    """
    def __init__(self, config):
        pass
spatial_register = keras.utils.register_keras_serializable("spatial_flow/reduction")


@spatial_register
class keras_neurons(keras.layers.Layer):
    """ 
    
    This class takes the initiative needed to turn a given keras layer into
    something which can be applied to a ND tensor in comparison format. 

    Support is provided per dimension for if one wants to tile the neuron over
    that dimension or share paramters instead. Additionally, so long as 
    the provided layer has a working from_config and to_config definition, 
    and can accept data in a (header, data, tail) format, it will work
    without trouble with this class. 

    The input for the class must be in comparison format to work. Additionally, do note
    that the last dimension of the comparison tensor is what will be reduced. As such,
    use tf.transpose to rearrange the tensor before and after operations if additional
    manual flexibility is needed.


    The primary initialization options are:
        
        - config   
        - layer_definition
        - tile_behavior
        - target_dim

    Defails
        - config: A valid spatial flow config.
        - layer_definitions:
            This can be a keras layer or serialized keras layer. It will be tiled according
            to the tiling rule in the tile_behavior parameter, and used to reduce target_dim of 
            the input kernel to a size of one. 

            This is not squeezed, allowing combinations.
        - tile_behavior:
            The other key part of the function, tile behavior governs how the keras layer provided for reducing
            the relationships is tiled across the input space. 

            For any given dimension of the spatial ND block, the layer can either be tiled across that dimension,
            producing per neuron differences between each entry along that dimension, or have the layer shared
            along that axis, producing a convolution on that dimension. 

            Three strings are predefined for convenience, and an additional mode is also provided allowing the manual specification
            per dimension of tiling behavior. 

                "Full" will fully tile along all dimensions. All dimensions will thus have independent parameters assigned to them
                "Partial" will tile only along the dimension corrolated with the target dimension. All the rest are treated as
                 convolution. This is the default
                "None" will not tile at all, and will simply apply a convolution to the target dimension.

            Additionally, if additional granularity is needed one may provide a bool list/tensor of length "spatial_length" where
            each bool entry indicates whether to tile along that dimension. True means tile, False means convoluti
        - target_dim: which entry of spatial_length needs to be reduced. Note that "partial" decides where to tile based off this.
    """

    def __init__(self, 
                 config, 
                 layer_definition, 
                 tile_behavior,
                 target_dim
    ):
      
        #Perform validation and broadcast steps. Do this by first defining reusable validation functions,

        pass
