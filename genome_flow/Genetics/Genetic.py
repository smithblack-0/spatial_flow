import tensorflow as tf




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
    @property
    def genome(self):
        return self._genome
    @property
    def gene(self):
        return self._gene
    @property
    def trait_templates(self):
        return self._genome.templates

    #define initializer
    def __init__(self, genome, name="Genetic", gene_catagory=None):
        # Begin with standard initialization
        super().__init__(name=name)

        # Check if there is already a gene for me
        if not genome.has_gene(name):
            # No gene found, make one.
            genome.new_gene(self, name, gene_catagory)
        # set the gene for the genetic layer.
        self._gene = genome.get_gene(name)
        self._genome = genome
        #prepare functions


    def add_trait(self, name, trait):
        if name not in self._gene.traits:
            self._gene.set_trait(name, trait)
        return self._gene.get_trait(name)