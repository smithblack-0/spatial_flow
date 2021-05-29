import tensorflow as tf
import tensorflow.keras as keras
from spatial_flow.config import spatial_config
import spatial_flow.utils.error_utils as error
import spatial_flow.utils.functions as functions

#setup registration function for keras serializer

spatial_register = keras.utils.register_keras_serializable("spatial_flow/core")




   
def indexed_broadcast(input, shape, indices):
    """ An estoteric variety of broadcast, the indexed
    broadcast allows for the assignment of specific 
    dimensions of the input shape to particular dimensions
    of the target broadcast shape.

    The input should be an input, a target broadcast shape,
    and a list of the same length as the input. The indices
    should each corrospond to one of the indices of shape.

    """
    #do some sanity testing

    tf.debugging.assert_integer(shape)
    tf.debugging.assert_integer(indices)
    tf.debugging.assert_rank(shape, 1)
    tf.debugging.assert_rank(indices, 1)

    #Rearrange for dimensions insertion.
    
   
    args = tf.argsort(indices)
   
    if(len(args) > 1):
        #if required because for some bizzare reason you cannot transpose a tensor back 
        #on top of itself. Bugfix
        output = tf.transpose(input, args)
    else:
        output = input
    #Insert dimensions

    for index in range(len(shape)):
        if(index not in indices):
            if(all(index < item for item in indices)):
                output = tf.expand_dims(output, axis=0)
            elif(all(index > item for item in indices)):
                output = tf.expand_dims(output, axis=-1)
            else:
                output = tf.expand_dims(output, axis=index)

    #perform broadcast and return result
    output = tf.broadcast_to(output, shape)
    return output




def unpack_standard(config, standard, callback, level="spatial", shape=None):
    """

    This function is dedicated to unpacking
    a standard tensor and then applying a callback
    function to each instance when finished. The result
    of the callback function is then restacked and the result
    is returned.

    The "level" parameter controls how far to unpack. The two options are "spatial" and "individual",
    unpacking respectively to the spatial domain and individual entries before calling on callback.
    Callback must return a tensor of shape "shape." If no shape is provided, it must return a tensor
    of the same shape as it's input.

    Support for ragged tensors is not provided.

    :param config: A valid spatial flow config
    :param level: either "spatial" or "individual"
    :param callback: A callback function to be called when unpacking
    :param shape: A tensor or tensorshape representing the expected output of the callback.
    :return: The restacked output
    """
    #perform verification
    if not isinstance(config, spatial_config): #config verify
        raise TypeError("unpack_standard: config was not a spatial_config")
    check = config.standard_config.is_standard(standard)
    if not check:
        raise TypeError("unpack_standard: %s" % check)
    if level not in ("spatial", "individual"):
        raise ValueError("unpack_standard: level was not either 'spatial' or 'individual'")
    if not callable(callback):
        raise TypeError("unpack_standard: callback was not function")
    if shape is None:
        shape = tf.TensorShape([None])
    shape = tf.TensorShape(shape)

    #calculate recursion threshold
    cfg = config.standard_config
    if(level == "spatial"):
        threshold = cfg.standard_rank - cfg.header_rank - cfg.tail_rank
    else:
        threshold = 1

    #Do transpose to batch form and setup restore permute
    permute = tf.concat(
        [cfg.header_indices, cfg.tail_indices, cfg.spatial_indices],
        axis=0)
    restore = tf.gather(cfg.standard_indices, permute, axis=0)
    restore = tf.concat([restore, tf.range(restore.shape[0], restore.shape[0] + shape.rank)], axis = 0)
    unpack_format = tf.transpose(standard, permute)

    #Setup expected return shape

    return_shape = unpack_format.shape[-threshold:].concatenate(shape)

    #run unpack
    repacked = unpacker(unpack_format, threshold, callback, return_shape)
    # Restore and return

    return tf.transpose(repacked, restore)


def unpack_reference(config, reference,  callback, shape=None, level="comparison",):
    """

    This function is dedicated to unpacking
    a reference tensor and then applying a callback
    function to each instance when finished. The result
    of the callback function is then repacked and the result
    is returned.

    The "level" parameter controls how far to unpack. The three options are "spatial", "comparison" and "individual",
    unpacking respectively to the spatial, comparison, and individual reference entries before calling on callback.
    Callback must return a tensor of shape "shape." If no shape is provided, the callback return
    must have rank 1.

    Support for ragged tensors is not provided.

    :param reference: the reference to be unpacked
    :param config: A valid spatial flow config
    :param level:  "spatial", "comparison, or "individual"
    :param callback: A callback function to be called when unpacking
    :param shape: A tensor or tensorshape representing the expected output of the callback.
    :return: The repacked output
    """
    # perform verification
    if not isinstance(config, spatial_config):  # config verify
        raise TypeError("unpack_reference: config was not a spatial_config")
    check = config.reference_config.is_reference(reference)
    if not check:
        raise TypeError("unpack_reference: %s" % check)
    if level not in ("spatial", "comparison", "individual"):
        raise ValueError("unpack_reference: level was not  'spatial', 'comparison', or 'individual'")
    if not callable(callback):
        raise TypeError("unpack_reference: callback was not function")
    if shape is None:
        shape = tf.TensorShape([None])
    shape = tf.TensorShape(shape)

    # calculate recursion threshold
    cfg = config.reference_config
    if level == "spatial":
        threshold = cfg.reference_rank
    elif level == "comparison":
        threshold = cfg.reference_rank - cfg.spatial_rank
    else:
        threshold = cfg.reference_rank - cfg.spatial_rank - cfg.comparison_rank
    # Do transpose to batch form and setup restore permute
    permute = tf.concat(
        [cfg.spatial_indices, cfg.comparison_indices, cfg.index_indices],
        axis=0)


    unpack_format = reference

    # run unpack
    repacked = unpacker(unpack_format, threshold, callback, shape)
    # Restore and return

    return repacked


def unpacker(tensor, threshold, callback, shape):
    """
    Unpacker is responsible for recursively
    unpacking a input tensor until the number of dimensions
    equals threshold. Each unpacked tensor is then
    called using callback.

    The return shape from callback must be the same as "shape."
    Unpacking proceeds from first to last.


    :param tensor: the tensor, state in recursion
    :param threshold: the threshold we are recursing to
    :param callback: the callback function
    :param shape: the shape of the callback output.
    :return: the repacked function
    """
    #perform basic sanity checking
    if not isinstance(shape, tf.TensorShape):
        raise TypeError("unpacker: Shape was not of type TensorShape")
    #Check if recursion has been met. If so, apply callback and return
    if(tensor.shape.rank is threshold):
        callback_result = callback(tensor)
        return callback_result
    #Recursion has not been met. Unpack another layer.

    output_shape = tensor.shape[1:-threshold].concatenate(shape)
    unpack_action = lambda input : unpacker(input, threshold, callback, shape)
    output = tf.map_fn(unpack_action, tensor, fn_output_signature=tf.TensorSpec(output_shape))
    return output

