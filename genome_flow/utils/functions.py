import tensorflow as tf



def broadcast_to(input, shape, source_dimension=None,  name=None):
    """ Broadcast to a shape
    The standard tensorflow broadcast function is sadly lacking 
    in certain options. It does not allow the specification
    of the dimension upon which broadcasting will start

    This forms something of a problem when one has header
    and tail sections.

    The additional parameter "source_index" tells this broadcast
    function where to begin the broadcast process. From this
    location, per the standard rules, the function works backwards and
    either the input and shape dimensions have to match, 
    or one of them must be one. After this, the original shape is restored

    If left as none, it defaults to the end.


    """
    #The standard tensorflow broadcast function is sadly lacking in options
    #




    #Setup default value
    print(shape)
    if(source_dimension == None):
        source_dimension = len(shape) -1 

    

    #Reshape into broadcast compatible format and construct permutation functions

    roll_directive = len(shape) - 1 - source_dimension
    permute = tf.range(len(shape))
    shape = tf.roll(shape, roll_directive, axis=0)
    permute = tf.roll(permute, -roll_directive, axis=0)

    #perform broadcast
    broadcast = tf.broadcast_to(input, shape)
   
    #permute back, return
    broadcast = tf.transpose(broadcast, permute, name=name)
    return broadcast 

