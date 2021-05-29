import tensorflow as tf
import tensorflow.keras as keras
import spatial_flow.utils.error_utils as error
import spatial_flow.core as core
"""

This section pertains to references and reference based functions


References are spatial blocks with spatial_length + 1 additional dimensions which are, in effect, 
sparse pointers to particular neurons in a standard spatial block. They are used for
initialization only, and are accepted by some sort of selector, with the most common being static. 

Their exact morphology is as follows. After the tail dimension, there is one additional dimension
whose length is equal to the number of spatial dimensions. This indexes the various mesh states.

Past this, there are spatial_length additional dimensions, with each additional dimension 
being capable of holding any number of pointers for that particular dimension.

References are extremely flexible, which is why the architecture is used. Of particular notice,
references naturally allow spatial flow to dynamically rearrange its architecture at the drop of the hat.



references are build using the reference language architecture 





"""


class Reference():
    """

    The base class for all references. This is designed to 
    be subclassed.

    Accepts one argument upon initialization. This argument is the configuration. 
    Upon being called, returns the identity reference. 

    Additionally, the "build" method will be provided with the index  of each
    given spatial coordinate, and will return a single comparison reference block.
    By default, it will be the identity case.

    If more complex behavior is desired, it is 
    possible to override this to get other functions.


    """

    def __init__(self, config):
        self._config = config
        self.reference_config = config.reference_config
    def __mesh(self, shape):

        tf.debugging.assert_integer(shape)
        tf.debugging.assert_rank(shape, 1)
        tf.debugging.assert_greater_equal(shape, 1)
        #This function makes a meshgrid for a given shape


        #Build meshgrid spatial creation instructions, then build meshgrid list

        mesh_instructions = [tf.range(item) for item in shape]
        mesh_list = tf.meshgrid(*mesh_instructions, indexing = "ij")
        mesh = tf.stack(mesh_list, -1) #reference index is at end

        return mesh



    def __run_build(self, build, input):
        #Perform the construction phase of call. Do this by
        #fetching the identity matrix, reshaping it, then providing it by
        #recursive function to build and stacking back up the result
        print(input)
        if(len(input.shape) == 1):
            #recursion is finished.

            #fetch comparsion block, validate it, then operate on it
            #under modulo to bring it into the correct range
            comparison_indices = build(input)

            #basic verification
            expected_dim =self.reference_config.spatial_rank + 1
            if(not tf.debugging.is_numeric_tensor):
                raise TypeError("Build output should be a numeric tensor, but it is not")
            tf.debugging.assert_integer(comparison_indices, "Build output should be integer, but was not")
            tf.debugging.assert_rank(comparison_indices, expected_dim, "Build output should have rank of %s, but did not" % expected_dim)
            
            #modulo operation
            
            comparison_indices = tf.unstack(comparison_indices, axis =-1)
            for i in range(self.reference_config.spatial_rank):
                comparison_indices[i] = tf.math.floormod(comparison_indices[i], self._config.spatial_shape[i])
            comparison_indices = tf.stack(comparison_indices, axis=-1)

            #return result

            return comparison_indices
        output = []
        for item in tf.unstack(input):
            output.append(self.__run_build(build,item))
        output = tf.stack(output, 0)
        return output

    def build(self, spatial_index):
        """ 

        build function.
        
        The build function is the location at which custom generation code 
        may be placed. Compare will be provided with a spatial index
        and thereafter may return a block of tensors.

        The return should be of length "spatial_length +1", where each spatial length
        is a comparison dimension tied to the associated spatial dimension,
        and the last dimension is the reference dimension, and of length "spatial_length"
       
        Additionally, the return will be modulod by the dimension length.
        Ragged tensors are not supported, so the same shape should be returned each
        time.

        The default configuration simply returns an identity case.

        """
        output = spatial_index
        for item in self._config.reference_config.spatial_indices:
            output = tf.expand_dims(output, axis = 0)
        return output

    def __call__(self):
        """ 
        
        The call function builds a spatial identity, 
        runs it through build, and broadcasts the result
        into a complete reference. 
        
        It then returns this.

        """

        config = self._config 

        #Build the spatial references. Do this by building the meshgrid,
        #then applying the build function.

        identity = self.__mesh(self._config.spatial_shape)
        reference = self.__run_build(self.build, identity)

        return reference



class Spatial_Kernel(Reference):
    """ 

    This class makes a spatial kernel reference 
    
    A spatial kernel reference is a reference defined in regards to 
    each spatial position and for which it is the case that a specially
    designed kernel is used to sample nearby spatial entries and return
    them as references..

    The kernel is centered upon the spatial location, and then
    acted upon by bias, delta, and shape. In the case that the kernel encounters the 
    edge of the space, it wraps around to the other edge.

    As always, the spatial case will be constructed and then 
    broadcast across the nonspatial dimensions.

    The primary initialization variables are 

        - config (duh)
        - kernel_shape
        - kernel_delta
        - kernel_bias
        - jusfification

    The details on these are:
        - config. A valid spatial config. 
        - kernel_shape:
            A 0D or 1D int tensor specifying the total area covered by the kernel along each 
            spatial dimension. If the tensor is 0D, it is broadcast to 1D and applied
            equally everywhere
        - kernel_delta:
            A 0D or 1D int tensor specifying how many indices are moved between each selection. In the 
            case of a 0D tensor, this is broadcast across the 1D spatial length.

            Default is 1
        - kernel_bias
            A 0D or 1D int tensor specifying how many indices the kernel will be shifted
            from it's justification position. Convention is the same as roll - positive
            shifts towards higher indices, negative shifts towards lower indices. A 
            0D tensor will be broadcast to all 

            Default is 0
        -justification
            A string or 1D list of strings which may be "center", "right", and "left"

            This defines how the kernel is centered on the spatial index. 
                "right": the kernel ends at the spatial index
                "left": the kernel begins at the spatial index
                "center" : the kernel is centered on the spatial index. If the kernel is odd, it is centered 
                    on the center left

            defaults to center

        """
    def __init__(self, config, kernel_shape, kernel_delta = 1, kernel_bias = 0, justification= "center"):
        #Begin verification step. Do this by first building reusable validation functions, then applying them. 
        #then handle the remaining cases. 
        super().__init__(config)
        def validate_integer(input, input_name, threshold):

            cfg_reference = self.reference_config
            #perform standard validation

            tf.debugging.assert_integer(input, "Input %s was not int" % input_name)
            tf.debugging.assert_rank_in(input,[0,1], "Input of %s did not have rank 0 or 1" % input_name)
           # input = tf.cast(input, dtype=tf.dtypes.int32)

            #broadcast if needed

            if(tf.rank(input) == 0):
                input = tf.broadcast_to(input, [cfg_reference.spatial_rank])
            else:
                if(not error.valid_lengths(input, [cfg_reference.spatial_rank])):
                   raise error.Dimensions_Error("Expected %s to be of length %s, was %s" %
                                               (input_name, cfg_reference.spatial_rank, len(input)))
            #finally, do threshold evaluation if requested

            if(threshold != None):
                tf.debugging.assert_greater_equal(input, threshold,
                                                 "Expected %s to be greater than or equal to %s, was not" 
                                                 %(input_name, threshold))
            return input
        def validate_string(input, input_name):

            #perform validation
            cfg_reference = self.reference_config
            input = error.cast(input, input_name, tf.dtypes.string, "string")
            if(not error.valid_dimensions(input, [0,1])):
                raise error.Dimensions_Error("Dimensions of %s not 0 or 1" % input_name)


            if(len(input.shape) == 0):
                input = tf.broadcast_to(input, [cfg_reference.spatial_rank])
            else:
                if(not error.valid_lengths(input, [config.spatial_rank])):
                   raise error.Dimensions_Error("Expected %s to be of length %s, was %s" %
                                               (input_name, cfg_reference.spatial_rank, len(input)))      
            [error.check_catagory(item, input_name, ["right", "left", "center"], "right, left, center") for item in input]
            return input

        kernel_shape = validate_integer(kernel_shape, "kernel_shape", 1)
        kernel_delta = validate_integer(kernel_delta, "kernel_delta", 1)
        kernel_bias = validate_integer(kernel_bias, "kernel_bias", None)
        justification = validate_string(justification, "justification")

        #Construct the change instructions. These will, for each index of each dimension, indicate something
        #to add to this index. the result will be operated on by modulo, wrapping the results around.


        right_range = lambda length : tf.range(length)
        left_range = lambda length : tf.range(-length+1, 1)
        center_range = lambda length : tf.range(-tf.math.floordiv(length,2), length-tf.math.floordiv(length, 2))

        def justification_case(case):
            if(case == "right"):
                return right_range
            if(case == "center"):
                return center_range
            if(case == "left"):
                return left_range

        justification_instruction = lambda index : justification_case(justification[index])(kernel_shape[index])
        bias_instruction = lambda index : tf.add(justification_instruction(index), kernel_bias[index])
        delta_instruction = lambda index : tf.multiply(bias_instruction(index), kernel_delta[index])

        change_instructions = []
        for index in range(self.reference_config.spatial_rank):
            instruction = delta_instruction(index)
            change_instructions.append(instruction)
        
        #The change instructions are made. Now, broadcast them to the 
        #correct shapes for application.

        comparison_shape = []
        for instruction in change_instructions:
            comparison_shape.append(len(instruction))
        comparison_shape = tf.constant(comparison_shape)
        for i in range(len(change_instructions)):
            change_instructions[i] = core.indexed_broadcast(change_instructions[i], comparison_shape, [i])
        
        change_instructions = tf.stack(change_instructions, -1)
        #Store them for use.

        self.change_instructions = change_instructions
    def build(self, input):
        #add the change instructions. Let broadcasting do the hard work.

        output = input
        output = tf.add(output, self.change_instructions)
        return output
    def get_config(self):
        pass



def identity_reference(config):
    """ Build a identify reference block. 

    Each reference corrosponds directly to the spatial position of the
    item in the block.

    Other dimensions are simply nonspatially broadcast.

    """
    return Reference(config)

def spatial_kernel(config, kernel_shape, kernel_delta = 1, kernel_bias = 0, justification = "center"):
    """ 

    This function makes a spatial kernel reference 

    A spatial kernel refernce is simply a series of references indexed 
    by additional dimensions of length 'spatial_length' and with 
    function defined widths, which have the effect of selecting 
    items according to a special spatially dependent kernel.

    The kernel is centered upon the spatial location, and then
    acted upon by bias, delta, and shape. In the case that the kernel encounters the 
    edge of the space, it wraps around to the other edge.

    As always, the spatial case will be constructed and then 
    broadcast across the nonspatial dimensions.

    The primary initialization variables are 

        - config (duh)
        - kernel_shape
        - kernel_delta
        - kernel_bias
        - jusfification

    The details on these are:
        - config. A valid spatial config. 
        - kernel_shape:
            A 0D or 1D int tensor specifying the total area covered by the kernel along each 
            spatial dimension. If the tensor is 0D, it is broadcast to 1D and applied
            equally everywhere
        - kernel_delta:
            A 0D or 1D int tensor specifying how many indices are moved between each selection. In the 
            case of a 0D tensor, this is broadcast across the 1D spatial length.

            Default is 1
        - kernel_bias
            A 0D or 1D int tensor specifying how many indices the kernel will be shifted
            from it's justification position. Convention is the same as roll - positive
            shifts towards higher indices, negative shifts towards lower indices. A 
            0D tensor will be broadcast to all 

            Default is 0
        -justification
            A string or 1D list of strings which may be "center", "right", and "left"

            This defines how the kernel is centered on the spatial index. 
                "right": the kernel ends at the spatial index
                "left": the kernel begins at the spatial index
                "center" : the kernel is centered on the spatial index. If the kernel is odd, it is centered 
                    on the center left

            defaults to center

        """

    #Begin verification step. Do this by first building reusable validation functions, then applying them. 
    #then handle the remaining cases. 

 




    #Construct and return the spatial kernel reference. 

    spatial_kernel = identity_reference(config)
    for index in range(config.spatial_length):
        
        #fetch needed parameters
        roll_instruction = roll_instructions[index]
        roll_axis = config.spatial_indices[index]

        #Construct and perform roll

        roll_directive = lambda index : tf.roll(spatial_kernel, roll_instruction[index], roll_axis)
        spatial_kernel = tf.map_fn(roll_directive, tf.range(len(roll_instruction)))

        #Transpose back to standard format

        spatial_kernel = tf.transpose(spatial_kernel, tf.roll(tf.range(len(spatial_kernel.shape)), -1, 0))

    return spatial_kernel
 

def region(config, region_location, region_shape):
    """

    The region reference produces a reference series
    for which all references at all spatial dimensions
    refer to the regional block indicated

    Not yet implimented.

    """

    pass


