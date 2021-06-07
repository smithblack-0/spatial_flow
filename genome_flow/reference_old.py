import tensorflow as tf
import tensorflow.keras as keras
import genome_flow.utils.error_utils as error
import genome_flow.core as core

"""

This section pertains to references




"""
class



class Old_Reference:
    """

    A reference is, in essence, a series of pointers located on
    a spatial-comparison grid each pointing to an individual
    spatial location, along with associated support properties
    providing information on mutability and other properties.

    By modifying, rewriting, and fetching references
    a variety of advanced behavior may be implimented.

    The property "reference" is the actual reference. It cannot
    be directly modified.

    The property "relative_reference" can be modified. It is instructions
    which point to nearby spatial-grid locations, relative to the
    current location.

    The property "identity" identifies the spatial
    grid locations. It, again, cannot be modified.

    Under standard conditions, one should use the "update" method to make
    changes and the "unpack" method to make selections.
    """

    # primary properties.
    @property
    def reference(self):
        """ Returns the current reference """
        return self._reference

    @property
    def relative_reference(self):
        """ Returns the relative reference """
        return self._relative_reference

    @relative_reference.setter
    def relative_reference(self, value):
        """

        Sets the relative reference directly.

        """

        # Sanity checking

        if not isinstance(value, tf.Tensor):
            raise TypeError("Expected 'reference' to be of type tf.Tensor. Instead was %s" % type(value))
        msg_shape_err = "Expected 'reference' to be of shape %s but was instead %s" % (
        self.reference_shape, value.shape)
        msg_int_err = "Expected 'reference' to be int tensor. Instead found %s" % value.dtype

        tf.debugging.assert_shapes([(value, self.reference_shape)], message=msg_shape_err)
        tf.debugging.assert_integer(value, message=msg_int_err)

        # set the relative reference

        self._relative_reference.assign(value)

        # set the true reference
        self._reference = self.__build_reference()

    @property
    def identity(self):
        return self._identity

    @property
    def mutable(self):
        return self._mutable

    @mutable.setter
    def mutable(self, value):
        """
        Setter for mutable

        :param value:
        :return:
        """

        tf.debugging.assert_shapes([(value, self.spatial_shape)],
                                   message="Expected mutable to be shape of spatial dimensions")
        tf.debugging.assert_integer(value, message="Expected mutable to be int")
        tf.debugging.assert_greater_equal(value, -1,
                                          message="Expected mutable to be greater than or equal to negative 1")
        tf.debugging.assert_less_equal(value, 1, "Expected mutable to be less than or equal to 1")

        # Update mutable. Do this by finding true pushes, false pushes, and applying them

        mutable_now_true = tf.where(tf.equal(value, 1))
        mutable_now_false = tf.where(tf.equal(value, -1))

        if mutable_now_true.shape[0] > 0:
            self._mutable = tf.tensor_scatter_nd_update(self._mutable, mutable_now_true,
                                                        [True] * mutable_now_true.shape[0])
        if mutable_now_false.shape[0] > 0:
            self._mutable = tf.tensor_scatter_nd_update(self._mutable, mutable_now_false,
                                                        [False] * mutable_now_false.shape[0])

    # config properties
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

    def __mesh(self, shape, dtype=tf.dtypes.int32):

        tf.debugging.assert_integer(shape)
        tf.debugging.assert_rank(shape, 1)
        tf.debugging.assert_greater_equal(shape, 1)
        # This function makes a meshgrid for a given shape

        # Build meshgrid spatial creation instructions, then build meshgrid list

        mesh_instructions = [tf.range(item) for item in shape]
        mesh_list = tf.meshgrid(*mesh_instructions, indexing="ij")
        mesh = tf.stack(mesh_list, -1)  # reference index is at end
        mesh = tf.cast(mesh, dtype)
        return mesh

    def __verify(self, value, name):

        """ Perform verification """
        tf.debugging.assert_rank(value, 1, message="%s rank was not 1" % name)
        tf.debugging.assert_integer(value, message="%s expected to be int, was not")
        tf.debugging.assert_greater_equal(value, 1,
                                          message="All of %s expected to be greater then or equal to one, was not")
        return value

    def __build_reference(self):
        # Using the stored relative reference, build the current
        # reference and store it.

        comparison = self.spatial_shape.rank
        index = comparison + self._comparison_shape.rank
        permute = tf.concat([tf.range(comparison, index), tf.range(0, comparison), tf.range(index, index + 1)], axis=0)

        add_form = tf.transpose(self.relative_reference, permute)
        add_form = tf.add(add_form, self.identity)
        modulo_form = tf.math.floormod(add_form, self.spatial_shape)
        output = tf.transpose(modulo_form, permute)
        return output

    def __init__(self, spatial_shape, comparison_shape):
        """
        The initialization method.



        :param spatial_shape:
        :param comparison_shape:
        """

        self._spatial_shape = tf.TensorShape(self.__verify(spatial_shape, "spatial_shape"))
        self._comparison_shape = tf.TensorShape(self.__verify(comparison_shape, "comparison_shape"))
        assert self._comparison_shape.rank == self.spatial_shape.rank, "comparison_shape and spatial_shape must have same rank"

        self._index_shape = tf.TensorShape([self.spatial_shape.rank])
        self._reference_shape = tf.TensorShape([*self.spatial_shape, *self.comparison_shape, *self._index_shape])

        # set up internal flatten numbers

        self._restore_spatial = self.spatial_shape
        self._flatten_spatial = tf.TensorShape([tf.reduce_prod(self.spatial_shape)]).concatenate(self.comparison_shape).concatenate(self.index_shape)
        self._flatten_identity = tf.TensorShape([tf.reduce_prod(self.spatial_shape)]).concatenate(self.index_shape)
        # set up initial configuration

        self._identity = self.__mesh(self.spatial_shape)
        self._relative_reference = tf.Variable(tf.zeros(self.reference_shape, tf.dtypes.int32))
        self._mutable = tf.fill(self.spatial_shape, True)
        self._reference = self.__build_reference()
    def update(self, callback):
        """"

        This function is dedicated to unpacking a reference to the comparison level, then feeding the callback function with each unpacked reference.
        It should be the primary method by which references are modified.  The update function will provide each mutable comparison reference to the
        callback function for modification. Nonmutable references are excluded.

        Callback must be a function which accepts as it's input first a comparison-reference, and then a spatial-index. It must then
        return a dict with entries "reference" and "mutable". "reference"  must be a tensor of comparison-reference type, and "mutable" can
        be "True" or "False", with True allowing further changes and false preventing them.\

        Callback should return the relative position to a nearby spatial position of interest.

        :param callback: A function which
        """


        if not callable(callback):
            raise TypeError("Reference - unpack_reference: callback was not a function")
        shape = self.comparison_shape.concatenate(self.index_shape)

        def callback_wrapper(unpacked, spatial_index):

            # Test if user code even runs. If not, raise reason.
            try:
                output = callback(unpacked, spatial_index)
            except Exception as err:
                msg = "Error in user callback function - %s" % err
                raise ValueError(msg) from err
            #Check if dict has the right keys

            if not isinstance(output, dict):
                raise TypeError("Error in user callback - output was not dict")
            if "reference" not in output.keys():
                raise ValueError("Error in user callback - did not return dict with 'reference' key")
            if "mutable" not in output.keys():
                raise ValueError("Error in user callback - did not return dict with 'mutable' key")
            if not isinstance(output["reference"], tf.Tensor):
                raise TypeError(
                    "Error in user callback function - return was not none or tensor, but %s" % type(output))


            #Assert sane entries.
            tf.debugging.assert_shapes([(output["reference"], shape)],
                                       message="Error in user callback function. Return was not " +
                                               "of shape %s or None" % shape)
            tf.debugging.assert_type(output["reference"],
                                     unpacked.dtype,
                                     message="Error in user callback function. " +
                                             "Return was of dtype %s but reference was of dtype %s"
                                             % (output["reference"].dtype, unpacked.dtype))
            tf.debugging.assert_type(output["mutable"], tf.dtypes.bool, "Error in user callback function, mutable not bool")
            #Return result
            return output

        #reshape items, exclude mutable

        reshaped_ref = tf.reshape(self.relative_reference, self._flatten_spatial)
        reshaped_identity = tf.reshape(self.identity, self._flatten_identity)
        reshaped_mut = tf.reshape(self._mutable, [-1])
        excluded_ref = tf.boolean_mask(reshaped_ref, reshaped_mut)
        excluded_identity = tf.boolean_mask(reshaped_identity, reshaped_mut)
        #run map
        output_sig = {"reference" : tf.TensorSpec(shape, self.reference.dtype),
                      "mutable" : tf.TensorSpec(None, tf.dtypes.bool) }
        map_func = lambda index : callback_wrapper(excluded_ref[index], excluded_identity[index])
        map_range = tf.range(0, excluded_ref.shape[0])
        map_ref = tf.map_fn(map_func, map_range, fn_output_signature=output_sig)
        mutables = map_ref["mutable"]
        references = map_ref["reference"]

        #update references
        restored_ref = tf.tensor_scatter_nd_update(reshaped_ref, tf.where(reshaped_mut), references)
        restored_mut = tf.where(tf.tensor_scatter_nd_update(reshaped_mut, tf.where(reshaped_mut), mutables), 0, -1)
        self.relative_reference = tf.reshape(restored_ref, self.reference_shape)
        self.mutable = tf.reshape(restored_mut, self.spatial_shape)

        return self
    def unpack(self, callback, shape, dtype=tf.dtypes.float32):
        """

        This function is dedicated to unpacking the reference to the callback level, then feeding the callback function with each unpacked
        callback reference. Some sophisticated math is meant to keep it fast. It should be the primary interface
        between a reference and the world.

        callback should accept a comparison-reference and return a tensor the shape of "shape"

        :param callback: A callback function to be called when unpacking. Should accept one paramter representing the unpacked shape
        :param shape: A tensor or tensorshape representing the expected output of the callback..
        :param update: Whether the result should be pushed into the reference tensor. If not, result is instead returned.
        :return: The repacked output
        """
        # perform validation

        if not callable(callback):
            raise TypeError("Reference - unpack_reference: callback was not a function")
        if shape is None:
            raise TypeError("Reference - unpack - must provided a shape")
        shape = tf.TensorShape(shape)

        # wrap callable in error checking
        tf.print("shape %s " % shape)

        def callback_wrapper(unpacked):

            # Test if user code even runs. If not, raise reason.
            try:
                output = callback(unpacked)
            except Exception as err:
                msg = "Error in user callback function - %s" % err
                raise ValueError(msg) from err

            # Test if output is valid. If not, intercept and elaborate

            if not isinstance(output, tf.Tensor):
                raise TypeError(
                    "Error in user callback function - return was not none or tensor, but %s" % type(output))
            tf.print(output)
            tf.debugging.assert_shapes([(output, shape)],
                                       message="Error in user callback function. Return was not " +
                                               "of shape %s or None" % shape)
            tf.debugging.assert_type(output,
                                     dtype,
                                     message="Error in user callback function. " +
                                             "Return was of dtype %s but reference was of dtype %s"
                                             % (output.dtype, dtype))
            return output


        #begin performing reshape to comparison dimensions. Then go ahead and
        #remove nonmutable entries


        #reshape, run map, and restore

        reshaped_ref = tf.reshape(self.reference, self._flatten_spatial)
        mapped_out = tf.map_fn(callback_wrapper, reshaped_ref, fn_output_signature=tf.TensorSpec(shape, dtype))
        restored = tf.reshape(mapped_out, self._restore_spatial.concatenate(shape))

        # return result
        return restored



class Reference_Op():
    """

    This class is designed to allow the easy and tearless modification and modification of references
    It possess a single method designed to be overridden, "modify", into which code can be placed to
    make modifications at the comparison or index level, and upon being called with a reference
    will apply modify in the appropriate fashion with a few details defined in init

    See modify for more details
     """
    def __init__(self, mutable=True):
        pass

    def __run_modify(self, comparison_indices, spatial_indices):
        """
        Run_modify is responsible for managing and error checking modify.

        :param input:
        :return: The reference
        """


        try:
            output = self.modify(comparison_indices, spatial_indices)
        except Exception as err:
            msg = "Reference_Op - error in modify: %s" % err
            raise Exception(msg)
        #perform basic sanity checking
        if not isinstance(output, (list, tuple)):
            raise TypeError("Reference_Op - error in modify - Expected return of list or tuple")
        if len(output) != 2:
            raise ValueError("Referenoce_Op - error in modify - list length was not 2")

        return {"reference" : output[0], "mutable" : output[1]}


    def modify(self, comparison_indices, spatial_index):
        """

        The modify function is designed to be overwritten, and may be utilized
        as follows.

        when the reference op is called with a reference, the reference willl
        be unpacked to the comparison_indices level where the reference is mutable.
        This, along with each spatial_index related, will then be passed into modify.

        Modify is then expected to return a list containing the new relative comparison_indices
        and a bool in the second entry. True means it remains mutable, false means it does not.

        :param comparison_indices: the unpacked comparison indices
        :param spatial_index: The location of the comparison_indices on the spatial grid
        :return a list containing first the tensor and second the mutability bool.
        """

    def __call__(self, reference):
        """

        call this with a reference, and use it to update it

        """

        #error check

        if not isinstance(reference, Reference):
            raise TypeError("Reference_Op - did not recieve input of type 'reference' on call")

        #Run

        return reference.update(self.__run_modify)



def spatial_kernel(reference, kernel_delta=1, kernel_bias=0, justification="center", mutable=True):
    # Begin verification step. Do this by first building reusable validation functions, then applying them.
    # then handle the remaining cases.
    def validate_integer(input, input_name, threshold):

        # perform standard validation

        tf.debugging.assert_integer(input, "Input %s was not int" % input_name)
        tf.debugging.assert_rank_in(input, [0, 1], "Input of %s did not have rank 0 or 1" % input_name)
        # input = tf.cast(input, dtype=tf.dtypes.int32)

        # broadcast if needed

        if (tf.rank(input) == 0):
            input = tf.broadcast_to(input, [reference.comparison_shape.rank])
        else:
            if (not error.valid_lengths(input, [reference.comparison_shape.rank])):
                raise error.Dimensions_Error("Expected %s to be of length %s, was %s" %
                                             (input_name, reference.comparison_shape.rank, len(input)))
        # finally, do threshold evaluation if requested

        if (threshold != None):
            tf.debugging.assert_greater_equal(input, threshold,
                                              "Expected %s to be greater than or equal to %s, was not"
                                              % (input_name, threshold))
        return input

    def validate_string(input, input_name):

        # perform validation
        input = error.cast(input, input_name, tf.dtypes.string, "string")
        if (not error.valid_dimensions(input, [0, 1])):
            raise error.Dimensions_Error("Dimensions of %s not 0 or 1" % input_name)

        if (len(input.shape) == 0):
            input = tf.broadcast_to(input, [reference.comparison_shape.rank])
        else:
            if (not error.valid_lengths(input, [reference.comparison_shape.rank])):
                raise error.Dimensions_Error("Expected %s to be of length %s, was %s" %
                                             (input_name,reference.comparison_shape.rank, len(input)))
        [error.check_catagory(item, input_name, ["right", "left", "center"], "right, left, center") for item in
         input]
        return input

    kernel_delta = validate_integer(kernel_delta, "kernel_delta", 1)
    kernel_bias = validate_integer(kernel_bias, "kernel_bias", None)
    justification = validate_string(justification, "justification")

    # Construct the change instructions. These will, for each index of each dimension, indicate something
    # to add to this index. the result will be operated on by modulo, wrapping the results around.

    right_range = lambda length: tf.range(length)
    left_range = lambda length: tf.range(-length + 1, 1)
    center_range = lambda length: tf.range(-tf.math.floordiv(length, 2), length - tf.math.floordiv(length, 2))

    def justification_case(case):
        if (case == "right"):
            return right_range
        if (case == "center"):
            return center_range
        if (case == "left"):
            return left_range

    justification_instruction = lambda index: justification_case(justification[index])(reference.comparison_shape[index])
    bias_instruction = lambda index: tf.add(justification_instruction(index), kernel_bias[index])
    delta_instruction = lambda index: tf.multiply(bias_instruction(index), kernel_delta[index])

    change_instructions = []
    for index in range(reference.comparison_shape.rank):
        instruction = delta_instruction(index)
        change_instructions.append(instruction)

    # The change instructions are made. Now, broadcast them to the
    # correct shapes for application.

    comparison_shape = []
    for instruction in change_instructions:
        comparison_shape.append(len(instruction))
    comparison_shape = tf.constant(comparison_shape)
    for i in range(len(change_instructions)):
        change_instructions[i] = core.indexed_broadcast(change_instructions[i], comparison_shape, [i])

    change_instructions = tf.stack(change_instructions, -1)
    tf.print(change_instructions)
    # Go apply the instructions
    update = lambda i, j : {"reference" : change_instructions, "mutable" : mutable}
    return reference.update(update)

