import tensorflow as tf
import tensorflow.python.framework.errors as errors

"""

Errors Utility file

This contains the error utility functions which are commonly found for input handling, sanity checking,
and sanitation throughout the program. The location is centralized to minimize needed tweaking and follow
DRY coding practices, and each function is also protected against the programmer him/herself making
boneheaded mistakes

This is meant to be a private file.

"""


""" Cast section """

def cast(input, input_name, dtype, dtype_name):
    """ cast tool

    The cast tool attempts to cast a function into a format,
    and throws a meaningful error message about the input
    if it cannot

    raises TypeError if cast fails
    raises UtilityError if utility parameters are insane


    """

    #Check the sanity of the utility paramters
    if(not isinstance(dtype, tf.dtypes.DType)):
       raise Utility_Error("cast input 'dtype' did not receive type 'tensorflow.dtype'", dtype)
    if(not isinstance(input_name, str)):
       raise Utility_Error("'cast' input 'input_name' did not receive type 'str'", input_name)
    if(not isinstance(dtype_name, str)):
       raise Utility_Error("'cast' input 'dtype_name' did not receive type 'str'", dtype_name)
   #Attempt the cast

    try:
        return tf.cast(input, dtype)
    except errors.UnimplementedError as err:
        msg = "Cast failed for input %s to type %s: %s" % (input_name, dtype_name, err)
        raise TypeError(msg) from err
 



""" Conditions section """


def check_conditition(input, input_name, threshold, condition, condition_name):
    """

    Verifies that a binary condition from input "name" and with operator "condition" relative to "threshold"
    is satisfied. 

    """

    #Check the sanity of the utility parameters
    if(not isinstance(input_name, str)):
        raise Utility_Error("'check_condition' input 'name' did not receive input of type 'str'")
    if(not isinstance(condition_name, str)):
        raise Utility_Error("'check_condition' input 'condition_name' did not receive input of type 'str'")

    #Check the condition sanity. Also, catch additional utiltity parameters.
    if(not tf.reduce_all(condition(input, threshold))):
        raise errors.InvalidArgumentError(None, None, "Invalid input of name %s: %s is not %s %s" % (input_name, input_name, condition_name, threshold))
    return input

def greater_equal(input, input_name, threshold):
    """ Check if input is greater than threshold. If not, throw a meaningful error message """
    return check_conditition(input, input_name, threshold, tf.greater_equal, "greater than or equal to")

def greater(input, input_name, threshold):
    """ Check if input is greater than threshold. If not, throw meaningful error message """
    return check_conditition(input, input_name, threshold, tf.greater, "greater than")


def equal(input, input_name, threshold):
    """ Check if input is equal to threshold. If not, throw meaningful error message """
    return check_conditition(input, input_name, threshold, tf.equal, "equal to")

def less_equal(input, input_name, threshold):
    """ Check if input is less than or equal to threshold. If not, throw meaningful error message"""
    return check_conditition(input, input_name, threshold, tf.less_equal, "less than or equal to")
def less(input, input_name, threshold):
    """ Check if input is less than threshold. If not, throw meaningful error message """
    return check_conditition(input, input_name, threshold, tf.less, "less than")



"""  Catagories section """

def check_catagory(input, input_name, catagories, catagories_labels):
    """ Check if input lies within one of the given catagories. 
    
    If utility parameters are insane, raises a utility error.
    Else, for type and catagory problems respectively:

    "Input of name 'input_name' has invalid type. Type is not 'type_labels'
    "Input of name 'input_name' is not valid: Must be 'catagories_labels'
    
    """

    #Check that utility paramaters are sane.
    if(not isinstance(catagories, (list, tuple))):
        raise Utility_Error("'check_catagory' input 'catagories' did not receive input of type 'list' or 'tuple'", catagories)
    if(not isinstance(catagories_labels, str)):
        raise Utility_Error("'check_catagory' input 'catagories_labels' did not receive input of type 'str", catagories_labels)
    if(not isinstance(input_name, str)):
        raise Utility_Error("'check_catagory' input 'input_name' did not receive input of type 'str'",input_name )

    #Check that catagory is satisfied. 

    if(input not in catagories):
        raise ValueError("Input of name %s is not valid: Must be %s" % (input_name, catagories_labels))
    return input

""" Dimensions and lengths """

def valid_dimensions(input, dimensions):
    """ This functions checks if a tensorflow tensor has legal dimensions 
    
    Dimensions should be a list of ints, each representing a dimension which is allowed.

    If the dimensions of the input tensor is not in the shape, return false
    
    """
  
    #check that dimensions are valid
    if(not tf.is_tensor(input)):
        raise Utility_Error("check_dimensions input was not tensor")
    if(not isinstance(dimensions, (list, tuple))):
        raise Utility_Error("check_dimensions 'dimensions' was not a list or tuple")
    if(len(input.shape) not in dimensions):
        return False
    return True

def valid_lengths(input, lengths):
    """ This function checks if a tensorflow tensor has a legal length 
    
    if not, it returns false. Else, it returns true
    """
    if(not tf.is_tensor(input)):
        raise Utility_Error("valid_length input was not a tensor")
    if(not isinstance(lengths, (list, tuple))):
        raise Utility_Error("'valid_length' input 'lengths' was not a tuple or list")
    if(len(input) not in lengths):
        return False
    return True

class Dimensions_Error(Exception):
    def __init__(self, msg, arg=None):
        msg = "Dimensions_Error: " + msg
        super().__init__(msg)
        self.prb = arg
        
class Utility_Error(Exception):
    def __init__(self, msg, arg=None):
        msg = "Utility Error: " + msg
        super().__init__(msg)
        self.prb = arg

class Reference_Error(Exception):
    def __init__(self, msg, arg=None):
        msg = "Reference Error: " + msg
        super().__init__(msg)
        self.prb = arg

class Selection_Error(Exception):
    def __init__(self, msg, arg=None):
        msg = "Selection Error: " + msg
        super().__init__(msg)
        self.prb = arg