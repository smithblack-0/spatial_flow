



class synapse_template:
    """
    A synapse template consists of the
    definitons defining a relative synapse
    region along with the associated reduction
    class

    Interestingly enough, the majority of the
    learning takes place in the synapses.

    Synapse templates possess the special method
    instance_template, which as the name suggest
    builds a synapse attached to a particular neuron.

    It is the method that users can be expected to interact with.

    """
    def __init__(self, relative_location, reducer):

    def build(self):
    def instance_template(self):

class synapse:
    """


    A synapse consists of the definitions and
    methods needed to successfully query
    a region of embedding space for the
    entries of all contained neurons

    The synapse class serves as something of a template
    for synapses.
    It consists of three things.

    One is the definition as to the shape
    and location of the region examined. This
    is what must be defined when first initialized.
    The location will always be given relative to
    a neural location

    Two is a method allowing the registering
    of a actual neuron


    """
