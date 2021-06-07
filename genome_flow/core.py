import tensorflow as tf
import tensorflow.keras as keras
import genome_flow.utils.error_utils as error
import genome_flow.utils.functions as functions

#setup registration function for keras serializer

spatial_register = keras.utils.register_keras_serializable("genome_flow/core")



   
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
    output = tf.map_fn(unpack_action, tensor, fn_output_signature=tf.TensorSpec(output_shape, tensor.dtype))
    return output

