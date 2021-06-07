


class Neuron_Error(Exception):
    """

    Neuron Error Class

    """
    def __init__(self, msg):
        msg = "Neuron Error: " + msg
        super().__init__(msg)

class MutatorError(Exception):
    """

    Mutator error class

    """
    def __init__(self, msg):
        msg = "Mutator Error: " + msg
        super().__init__(msg)

class GeneticError(Exception):
    """

    Error for genome problems

    """