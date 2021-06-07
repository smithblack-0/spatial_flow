import tensorflow as tf


class Neural_Space:
    """
    The Neural Space base class

    Neurons are embedded in a multidimensional
    space.
    """
    def __init__(self, name, dimensions, neurons):
        #Ensure parameters are sane

        assert isinstance(name, str), "Input name was not string"
        assert tf.debugging.is_numeric_tensor(dimensions), "Input 'dimensions' was not a numeric tensor"
        tf.debugging.assert_greater(dimensions, 0)

        #store data, initialize neurons

        self._name = name
        self._dimensions = dimensions


    #construction methods
    def build(self, input_shape):
        """

        Take all the templates and
        convert them into actual neurons.

        Do this by going through each
        template in comparison to every other
        template and see what regions overlap

        """
        for neuron in self._neurons:
            #Outer loop. Considers a particular
            #case

            matching_neurons = []

            for inner_neuron in self._neurons:
                #Inner loop. Consider this case

                if neuron.is_in_regions(inner_neuron.location):
                    matching_neurons.append(inner_neuron)
            neuron.build_connections(matching_neurons)


class Neuron:
    """
    The base Neuron Class

    """
    def __init__(self, neural_space, location, combiner, regions=None):
        self._environment = neural_space
        self._combiner = combiner
        if regions is None:
            self._regions = {}
        else:
            self._regions = regions
        self._state = 0




