import tensorflow as tf


class Spatial_State(tf.keras.layers.Layer):
    """

    The spatial state class is the entity responsible for managing and tracking
    neural states, as well as performing nonspatial unpacking when so requested

    It can be initialized with any Keras initializer or even a custom
    initializer, and can be modified directly through the state property.
    """
    @property
    def state(self):
        return self._state
    @property
    def batch_dims(self):
        return self._batch_dims
    @property
    def spatial_dims(self):
        return self._spatial_dims
    @property
    def channel_dims(self):
        return self.
    @property
    def total_dims(self):
        return self._total_dims
    def __init__(self, spatial_dims, channel_dims=[], batch_dims=[None], initialization ="zeros"):
        """_channel_dims

        The initialization method.

        Remarkably little actual initialization is performed here, as such actions must wait upon
        the spatail state being registered to a unit and fed from a interface which knows the batch
        shape

        :param spatial_dims: A 1D list of integers specifying the spatial dimensions. Must be integers
        :param channel_dims: A 1D tensorshape, which may have none as entries, corrolated to any tail parameters such as
            channels
        :param batch_dims: A 1D tensorshope, which may have none as entries, corrolated to any batch parameters
        :param initialization: What form to initialize any tensor to.
        """
        super().__init__(False)

        self._batch_dims = tf.TensorShape(batch_dims)
        self._spatial_dims = tf.TensorShape(tf.constant(spatial_dims))
        self._channel_dims = tf.TensorShape(channel_dims)
        self._total_dims = self._batch_dims.concatenate(self._spatial_dims).concatenate(self._channel_dims)

        if type(initialization) is str:
            intialization = tf.keras.initializers.get(initialization)

        self._state = tf.keras.Input(type_spec=tf.TensorSpec(self._total_dims))
    def call(self, null):
        return self.state