import tensorflow as tf
from genome_flow.errors import GeneticError
from genome_flow.Genetics.gene import gene



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
    @property
    def name(self):
        return self._name
    @property
    def statistics(self):
        return self._statistics
    def __init__(self, name, genes=None, statistics=None):
        self._name = name
        if genes is not None:
            self._genes = genes
            self._statistics = statistics
        else:
            self._genes = {}
            self._statistics = {"clones" : 0, "mutates" : 0, "crossbreeds" : 0}
    #


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
        return self._genes[name]
    def has_gene(self, name):
        if name in self._genes:
            return True
        return False

    def get_gene(self, name):
        if not self.has_gene(name):
            raise ValueError("Gene %s does not exist" % name)
        return self._genes[name]

    def add_template(self, trait):


    #Copying and Modification methods.
    def clone(self):
        """
        return a independent clone
        of this genome

        :return: another genome
        """
        cloned_genes = {}
        for key in self._genes:
            cloned_genes[key] = self._genes[key].clone()
        self._statistics["clones"] =+ 1
        return genome(self.name, cloned_genes, self.statistics)
    def mutate(self, number):
        """

        Create n copies of this genome, and mutate each copy

        :return: the mutated genome.
        """
        mutations = []
        for index in number:
            mutation = self.clone()
            for key in self._genes:
                mutation._genes[key] = self._genes[key].mutate()
            mutations.append(mutation)
    @staticmethod
    def crossbreed(genomes, number):
        """

        Crossbreed a series of genomes.

        This consists of taking the genomes, unpacking the genes for each genome and,
        at random, including a gene from a particular genome.

        :param genomes:
        :param Number: The number of offspring to make
        :return:
        """
        pass

`