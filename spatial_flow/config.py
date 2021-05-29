import tensorflow as tf
import tensorflow.keras as keras
import spatial_flow.utils.error_utils as error


spatial_register = keras.utils.register_keras_serializable("spatial_flow/config")


@spatial_register
class spatial_config(keras.layers.Layer):

    """ 
    
    The config object stores important configuration information in one 
    central, thouroughly error checked, location. 

    The config object exposes the following direct properties and important methods.

    Direct properties, settable:

    - standard_header_shape, a 1d int tensor telling the dimensions of header dimensions, such as batch size
    - standard_spatial_shape, a 1d int tensor telling the dimensions of the spatial block.
    - standard_tail_shape, a 1d int tensor telling the dimensions of the tail dimensions.

    Important methods:

    is_standard(input) - returns true if standard, and problem if not
    is_reference(input) - returns true if reference, and problem if not
    is_comparison(input) - returns true if comparison, and problem if not.

    Everything else is contained in individual config objects. The properties are:

    standard_config - an object representing the standard tensor representation. 
    reference_config - an object representing the reference tensor representation. 
    comparison_config - an object representing the comparison tensor representation.



    Ranks: Information about the ranks of the different objects utilized

    header_rank - rank of header_shape
    spatial_rank - rank of spatial shape
    tail_rank - rank of tail shape
    standard_rank - rank of standard shape
    comparison_rank - rank of comparison shape
    reference_rank - rank of reference shape.

    Shapes: Tensors and tuples indicating the shape and lengths of various dimensions
    
    standard_shape - the composite of header_shape, spatial_shape, tail_shape
    comparison_shape - shape comparisons should be. Correct rank, none located where length is uncertain.
            Tuple, not tensor!
    reference_shape- - shape references should be. Correct rank, none located where length is uncertain.
            Tuple, not tensor!
    
    Indices: a series of objects. Each object corrosponds to a specification 

    header_indices - the indices of header_shape in total_shape, useful for constructing permutations.
    spatial_indices - the indices of spatial_shape in total_shape, useful for constructing permutations.


    Shapes are usable directly with tf.debugging.assert. Additionally, where not listed as a tuple,
    one could in theory use them for casting as well.



    tail_shape - the indices of tail_shape in total_shape, useful for constructing permutations

    a spatial block, such as the spatial dimensions, header dimensions
    and tail dimensions,
   
    
    
    """
    #Define defaults

    _header_shape = tf.constant([], tf.dtypes.int32)
    _spatial_shape = tf.constant([], tf.dtypes.int32)
    _tail_shape = tf.constant([], tf.dtypes.int32)

    #Define initialization

    def __init__(self,
                    header_shape,       
                    spatial_shape,                    
                    tail_shape = None,
                    name = None,
                    ):
        super().__init__(name)
        self.header_shape = header_shape
        self.spatial_shape = spatial_shape
        self.tail_shape = tail_shape


    #Define settable properties

    @property
    def header_shape(self):
        return self._header_shape
    @property
    def spatial_shape(self):
        return self._spatial_shape
    @property
    def tail_shape(self):
        return self._tail_shape
    @header_shape.setter
    def header_shape(self, value):

        if(value != None):
           #Error check

            value = self.__verify(value, "header")

            #Commit

            self._header_shape = value
    @spatial_shape.setter
    def spatial_shape(self, value):
        if(value != None):
            #Error check

            value = self.__verify(value, "spatial")

            #commit

            self._spatial_shape = value
    @tail_shape.setter
    def tail_shape(self, value):
        if(value != None):
            #Error check

            value = self.__verify(value, "tail")

            #Commit

            self._tail_shape = value
    #define object properties
    @property
    def standard_config(self):
        return standard(self._header_shape, self._spatial_shape, self._tail_shape)
    @property
    def reference_config(self):
        return reference(self._spatial_shape)
    @property
    def compare_config(self):
        return compare(self._header_shape, self._spatial_shape, self._tail_shape)

    #update functions

    def __verify(self, value, name):

        """ Perform verification """
        tf.debugging.assert_rank(value, 1, message = "%s rank was not 1" % name)
        if(len(value) != 0):
            tf.debugging.assert_integer(value, "%s rank was not integer, was %s" % (name, value))
        return value




    def get_config(self):
        return {"header_shape" : self.header_shape, "spatial_shape" : self.spatial_shape, "tail_shape" : self.tail_shape}


@spatial_register
class standard(keras.layers.Layer):
    """ the config file for the standard tensor 
    
    This is initialized automatically, and exposes properties:

    header_shape:    a 1d int tensor telling the dimensions of header dimensions, such as batch size
    spatial_shape:   a 1d int tensor telling the dimensions of the spatial block.
    tail_shape:      a 1d int tensor telling the dimensions of the tail dimensions.
    standard_shape:  the composite of header_shape, spatial_shape, tail_shape

    header_rank - rank of header_shape
    spatial_rank - rank of spatial shape
    tail_rank - rank of tail shape
    standard_rank - rank of standard shape

    header_indices - the indices of header_shape in standard_shape, useful for constructing permutations.
    spatial_indices - the indices of spatial_shape in standard_shape, useful for constructing permutations.
    tail_indices  - the indices corrosponding to the tail of standard_shape.
    standard_indices - all the indices of standard_shape

    It also has the method:

    is_standard(input):
        Checks if a tensor is in standard format matching the config.
        If it is, it returns True.

        If it is not, it returns a string explaining the problem.

    """

    def __init__(self, header_shape, spatial_shape, tail_shape):

        super().__init__()

        #Set shape properties up

        self._header_shape = tf.TensorShape(header_shape)
        self._spatial_shape = tf.TensorShape(spatial_shape)
        self._tail_shape = tf.TensorShape(tail_shape)
        self._standard_shape = tf.TensorShape([*header_shape, *spatial_shape, *tail_shape])

        #Set index properties up

        self._header_start = 0
        self._spatial_start = self.header_start + self.header_rank
        self._tail_start = self.spatial_start + self.spatial_rank


        self._header_indices = tf.range(self.header_start, self.spatial_start)
        self._spatial_indices = tf.range(self.spatial_start, self.tail_start)
        self._tail_indices = tf.range(self.tail_start, self.standard_rank)
        self._standard_indices = tf.range(self.header_start, self.standard_rank)


    #shape properties
    @property
    def header_shape(self):
        return self._header_shape
    @property
    def spatial_shape(self):
        return self._spatial_shape
    @property
    def tail_shape(self):
        return self._tail_shape
    @property
    def standard_shape(self):
        return self._standard_shape

    #rank properties
    @property
    def header_rank(self):
        return self._header_shape.rank
    @property
    def spatial_rank(self):
        return self._spatial_shape.rank
    @property
    def tail_rank(self):
        return self._tail_shape.rank
    @property
    def standard_rank(self):
        return self._standard_shape.rank

    #indices properties

    @property
    def header_indices(self):
        return self._header_indices
    @property
    def spatial_indices(self):
        return self._spatial_indices
    @property
    def tail_indices(self):
        return self._tail_indices
    @property
    def standard_indices(self):
        return self._standard_indices

    #start properties

    @property
    def header_start(self):
        return self._header_start
    @property
    def spatial_start(self):
        return self._spatial_start
    @property
    def tail_start(self):
        return self._tail_start

    #define functions
    def is_standard(self, input):
        #check if tensor
        if(not tf.is_tensor(input)):
            return "input was not tensor"
        #check if correct rank
        try:
            tf.debugging.assert_rank(input, self.standard_rank)
        except Exception:
            return "Input did not have rank %s, was instead %s" % (self.standard_rank, input.shape.rank)

        #check if correct shape
        try:
            tf.debugging.assert_shapes([(input, self.standard_shape)])
        except Exception:
            return "Shape did not match. Expected %s but got %s" % (self.standard_shape, input.shape)
        return True


    def get_config(input):
        config = {"header_shape" : self._header_shape,
                  "spatial_shape" : self._spatial_shape,
                  "tail_shape" : self._tail_shape
                  }
        return config


@spatial_register
class reference(keras.layers.Layer):
    """ 

    This object represents the reference configuration.

    is_reference(input):
        Check and see if the input is a reference. If not, return reason why as string.

    It possesses the following properties:

    spatial_shape  :   a shape tensor telling the dimensions of the spatial block.
    comparision_shape : a shape tensor telling the dimensions of the comparison block
    index_shape: a shape tensor telling the dimensions of the index
    reference_shape: a shape tensorshape. None for the comparison dims

    spatial_rank: the rank of the spatial shape
    comparison_rank: the rank of the comparison shape. Same as spatial_rank
    index_rank: The rank of the index shape. Always 1.
    reference_rank: The rank of the whole package

    spatial_start
    comparison_start
    index_start
    
    spatial_indices: The indices of the spatial section
    comparison_indices: The indices of the comparison section
    index_indices: the index of the "index" section
    reference_indices: all the indices

    """

    def __init__(self, spatial_shape):
        super().__init__()

        #setup shapes

        self._spatial_shape = tf.TensorShape(spatial_shape)
        self._comparison_shape = tf.TensorShape([None] * len(spatial_shape))
        self._index_shape = tf.TensorShape(len(spatial_shape))
        self._reference_shape = tf.TensorShape([*self.spatial_shape, *self.comparison_shape, *self._index_shape])

        #setup indices

        self._spatial_start = 0
        self._comparison_start = self.spatial_start + self.spatial_rank
        self._index_start = self.comparison_start + self.comparison_rank

        self._spatial_indices = tf.range(self.spatial_start, self.comparison_start)
        self._comparison_indices = tf.range(self.comparison_start, self.index_start)
        self._index_indices = tf.range(self.index_start, self.reference_rank)
        self._reference_indices = tf.concat([self._spatial_indices, self._comparison_indices, self._index_indices], axis=0)

    #setup shape properties
    @property
    def spatial_shape(self):
        return self._spatial_shape
    @property
    def index_shape(self):
        return self._index_shape
    @property
    def comparison_shape(self):
        return self._comparison_shape
    @property
    def reference_shape(self):
        return self._reference_shape

    #rank properties
    @property
    def spatial_rank(self):
        return self._spatial_shape.rank
    @property 
    def comparison_rank(self):
        return self._comparison_shape.rank
    @property
    def index_rank(self):
        return self._index_shape.rank
    @property
    def reference_rank(self):
        return self._reference_shape.rank

    #index properties

    @property
    def spatial_start(self):
        return self._spatial_start
    @property
    def comparison_start(self):
        return self._comparison_start
    @property
    def index_start(self):
        return self._index_start

    @property
    def spatial_indices(self):
        return self._spatial_indices
    @property
    def comparison_indices(self):
        return self._comparison_indices
    @property
    def index_indices(self):
        return self._index_indices
    @property
    def reference_indices(self):
        return self._reference_indices
    #define functions

    def get_config(self):
        return {"spatial_shape" : self._spatial_shape}
    def is_reference(self, input):
        
        #check if tensor
        if(not tf.is_tensor(input)):
            return "Reference was not tensor"
        #check if tensor has the correct rank

        try:
            tf.debugging.assert_rank(input, self.reference_rank)
        except tf.errors.InvalidArgumentError:
            return "Reference had incorrect rank. Expected %s" % self.reference_rank
        #check if tensor is made up of integers
        try:
            tf.debugging.assert_integer(input)
        except tf.errors.InvalidArgumentError:
            return "Reference was not made up of integers"

        #check if tensor shape is sane

        try:
            tf.debugging.assert_shapes([(input, self.reference_shape)])
        except tf.errors.InvalidArgumentError:
            return "Reference had invalid shape"

        
        #check if index values within margins for each dimension 
      
        for i in range(self.spatial_rank):
            dimension_indices = input[..., i]
           # dimension_indices = tf.gather(input, i, -1) #slice out all the entries.
            try:
                tf.debugging.assert_greater_equal(dimension_indices, 0)
            except tf.errors.InvalidArgumentError:
                return "Reference had negative index in dimension %s" % i
            try:
                tf.debugging.assert_less(dimension_indices, self.spatial_shape[i])
            except:
                return "Reference had index greater than or equal to %s along dimension %s" % ( self.spatial_shape[i], i)
        #passed. Return true

        return True

@spatial_register
class compare(keras.layers.Layer):

    """
    This object represents the compare configuration.

    It has method:

    is_compare(input):
        Check and see if the input is in compare configuration. If not, return reason why as string, else reture true

    It possesses the following properties:

    header_shape : a 1d int shape telling the dimensions of the spatial block
    spatial_shape  :   a 1d int tensor telling the dimensions of the spatial block.
    comparison_shape: a shape telling information about the comparison block
    tail_shape : a 1d int shape telling the dimensions of the tail block.
    compare_shape: a tensorshape. 

    spatial_rank: the rank of the spatial shape
    comparison_rank: the rank of the comparison sector. Same as spatial_rank
    compare_rank: The rank of the whole package

    spatial_start
    comparison_start
    
    spatial_indices: The indices of the spatial section
    comparison_indices: The indices of the comparison section
    compare_indices: all the indices

    """

    def __init__(self, header_shape, spatial_shape, tail_shape):
        super().__init__()

        #define shapes

        self._header_shape = tf.TensorShape(header_shape)
        self._spatial_shape = tf.TensorShape(spatial_shape)
        self._tail_shape = tf.TensorShape(tail_shape)
        self._comparison_shape = tf.TensorShape([None] * self.spatial_shape.rank)
        self._compare_shape = tf.TensorShape([*header_shape, *spatial_shape, *tail_shape, *[None] * self.spatial_shape.rank])

        #define indices

        self._header_start = 0
        self._spatial_start = self.header_start + self.header_rank
        self._tail_start = self.spatial_start + self.spatial_rank
        self._comparison_start= self.tail_start + self.tail_rank
        
        self._header_indices = tf.range(self.header_start, self.spatial_start)
        self._spatial_indices = tf.range(self.spatial_start, self.tail_start)
        self._tail_indices = tf.range(self.tail_start, self.comparison_start)
        self._comparison_indices = tf.range(self.comparison_start, self.compare_rank)
        self._compare_indices = tf.range(self.header_start, self.compare_rank)

    #shape properties
    @property
    def header_shape(self):
        return self._header_shape
    @property
    def spatial_shape(self):
        return self._spatial_shape
    @property
    def tail_shape(self):
        return self._tail_shape
    @property
    def comparison_shape(self):
        return self._comparison_shape
    @property
    def compare_shape(self):
        return self._compare_shape

    #rank properties

    #rank properties
    @property
    def header_rank(self):
        return self._header_shape.rank
    @property
    def spatial_rank(self):
        return self._spatial_shape.rank
    @property
    def tail_rank(self):
        return self._tail_shape.rank
    @property
    def comparison_rank(self):
        return self._comparison_shape.rank
    @property
    def compare_rank(self):
        return self._compare_shape.rank

    #indices properties

    @property
    def header_start(self):
        return self._header_start
    @property
    def spatial_start(self):
        return self._spatial_start
    @property
    def tail_start(self):
        return self._tail_start
    @property
    def comparison_start(self):
        return self._comparison_start
    
    @property
    def header_indices(self):
        return self._header_indices
    @property
    def spatial_indices(self):
        return self._spatial_indices
    @property
    def tail_indices(self):
        return self._tail_indices
    @property
    def comparison_indices(self):
        return self._comparison_indices
    @property
    def compare_indices(self):
        return self._compare_indices

    #define functions:


    def get_config(self):
        return {"header_shape" : self.header_shape, "spatial_shape" : self.spatial_shape, "tail_shape" : self.tail_shape}
    def is_compare(self, input):
        #check if tensor
        if(not tf.is_tensor(input)):
            return "input was not tensor"
        #check if correct rank
        try:
            tf.debugging.assert_rank(input, self.compare_rank)
        except Exception:
            return "Input did not have rank %s" % self.standard_rank

        #check if correct shape
        try:
            tf.debugging.assert_shapes([(input, self.compare_shape)])
        except Exception:
            return "Shape did not match. Expected %s but got %s" % (self.standard_shape, input.shape)
        return True

