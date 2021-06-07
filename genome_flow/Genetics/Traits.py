import tensorflow as tf

class trait:
    """

    A trait is an object which represents a
    tensorflow variable and, possibly, a mutator

    It is a single item which can be modifed, and the
    lowest level of dynamic architecture.

    It contains two methods which must be well defined:
    Mutate, and Get.
    """

    @property
    def name(self):
        return self._name
    def __init__(self, name):
        """
        Define the initialization and place wrappers on location to remove
        unessicary parameters from user experience.

        :param trait_class:
        :param name:
        """
        self._name = name

    def get(self):
        """
        The get method. Used to actually return a trait instance

        :return: Whatever the trait is.
        :raise NotImplementedError: If it was never implimented
        """
        raise NotImplementedError("Method 'get' in trait % ")

    def mutate(self, *args, **kwargs):
        """

        The mutate method. Used to perform a mutation.

        Must be implemented, and can be provided with
        arbitrary variables from further up the stack.

        Take caution to ensure that your mutation
        is a COPY of the original.

        :return: the mutated trait
        """
        raise NotImplementedError("Method 'get' in trait % ")

class trait:

    """

    The base trait.

    A trait, at its most basic, consists of three things

    One is the actual trait itself.
    Two is a function or class, known as a "cloner",
    which when called returns an unconnected copy of a
    trait.

    Three is

    """

    def __init__(self, trait, cloner, mutator):
        self._trait = trait
        self._cloner = cloner
        self._mutator = mutator
    def clone(self):
        return self._cloner(trait)
    def mutate(self):
        return self._mutator(trait)
    def __call__(self):
        return self._trait
class Variable(trait):
    """

    The tensorflow variable trait is meant for storing and manipulating variables
    as traits. It expects to be provided a variable upon initialization.

    It does not have a mutate method attached, but may be provided with a standard
    mutator as an option. Alternatively, you can implement your own.

    """
    def __init__(self, variable, mutator=None):
        if not isinstance(variable, tf.Variable):
            raise TypeError("Expected tf variable")

        self._variable = variable
    def clone(self):
        initialization = {}


        initialization["initial_value"] = self._variable
        initialization["trainable"] = self._variable.trainable
        initialization["name"] = self._variable.name
        initialization["dtype"] = self._variable.dtype
        initialization["constraint"] = self._variable.constraint
        initialization["shape"] = self._variable.shape
        initialization["syncronization"] = self._variable.synchronization
        initialization["aggregation"] = self._variable.aggregation

        return tf.Variable(initialization)
    def mutate(self):
        self._clone