import tensorflow as tf
import tensorflow.keras as keras



class interface(keras.layers.Layer):
    """

    The ND interface.

    The ND interface class is designed to allow for the importation and exportation of data
    from a ND block in an organic and self-consistant manner.

    An ND interface consists of a provided definition of two ND blocks of differing (or similar) size, 
     a method of interface, the range of locations for the interface if relevent, and the number of neurons
     to interface on if relevant. Also provided is typically how interface updates are performed
     
    Generally, they can be divided into two types - random interlinks, which simply select N random neurons out of the 
    the two provided ND blocks within the restricted range and mark these as interface neurons then map them in the appropriate
    direction when called, or block interfaces, which slice N neurons out of a block maintaining shape if possible and use these
    for interlink calls. 

    For chaining and automation purposes, interlink can optionally return an interlink register, which can be fed into further
    interlinks to mark neurons as already defined for interfacing, and therefore off limits. 
    
    """
    pass

class block_interface(interface):
    """ 
    An ND block interface. 

    The ND block interface is designed to allow for the interconnection of large, contigous, regions of spatial states between
    different ND networks. It is useful for the importation or exportation of raw data into or out of a network in a specified format.
    For instance, Importing 2D images or 3D data into a network for processing, or exporting a 3D file for a statue.

    In general, the random interface should be used instead to interlink ND networks, after giving thought to where primary processing will occur and
    where conclusions should be drawn. 

    """
    pass

class random_interface(interface):

    """ 

    An ND random interface

    In mimicry of the organtic brain, the needed number of neurons for an interlink to be formed are
    randomly selected from between the larger block and the smaller block. With no neuron number
    provided, all neurons in the smaller block are assumed to need an interlink, otherwise only
    N neurons chosen at random are linked.


    """
    pass

class tensorflow_interface(interface):
    """

    The ND random interface is designed to allow the interlinking of normal
    tensorflow models to a ND spatial model with relevant ease. Given a provided
    tensorflow model width, a target interlink style, and and ND spatial model, 
    the interface will allow for information to be translated from a tensorflow
    model into a tensorflowND input.

    """

