import tensorflow as tf
from genome_flow import mutators
from genome_flow.errors import GeneticError


class Genetic(tf.keras.layers.Layer):
    """

    The base class of much of
    genome flow, this class wraps any
    "get_config" and "from_config" methods
    to, with proper preparation, seamlessly
    transition between keras and genome flow.

    A top level call to get_config will build a series of nested genes,
    and from_config will build a model based on said genome.
    """

    def __init__(self, genome, name="Genetic", category=None):
        # Begin with standard initialization
        super().__init__(name=name)

        # Check if there is already a gene for me
        if not genome.has_gene(name):
            # No gene found, make one.
            genome.new_gene(self, name, category)
        # set the gene for the genetic layer.
        self._gene = genome.get_gene(name)

    # Define the trait functions. These will set to the genome
    # if called and no trait is found, otherwise it will simply
    # return the value.

    def trait(self, name, cls, mutator, *args, **kwargs):
        if name in self._gene.traits:
            return self.trait(name)
        else:
            self._gene.add_trait(name, cls, mutator, args, kwargs)
            return self._gene.get_trait(name)


class genome:
    """
+
    The genome is the entity responsible for
    holding the entirety of the information
    regarding traits and mutations for a given
    model.

    It is capable of cloning itself and performing
    mutations with little issue. It may be fed into
    a genome flow stack to instance a model with a
    particular genome. It can be modified and instanced
    to perform any of a number of algorithms.

    The genome consists of an arbitrary number of genes,
    each of which may possess an arbitrary number of traits.
    Genes have both a unique name, along with a category to
    which they belong.
    """

    def __init__(self, name):
        self._name = name
        self._genes = {}

    def new_gene(self, name, category=None):
        """
        Make a new gene, belonging to the
        listed category

        :param name: name
        :param category:
        :return:
        """

        if category is None:
            category = "default"
        self._genes[name] = gene(name, category)

    def has_gene(self, name):
        if name in self._genes:
            return True
        return False

    def get_gene(self, name):
        if not self.has_gene(name):
            raise ValueError("Gene %s does not exist" % name)
        return self._genes[name]


class gene:
    """

    """
