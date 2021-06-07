
class gene:
    """
    A gene.

    A gene is a collection of traits relating
    to a Genetic object, and the logic needed to build
    and fetch said traits. A trait is, itself, a tensorflow
    variable that is watched by genome flow for modification
    purposes.

    A trait is an item which represents a individual piece of
    the genetic code of an Architecture. It is something which
    can be modified between rounds, if the appropriate mutator
    is assigned.

    Traits must always be assigned to a trait_class, which
    can then be modified by attaching  a matching mutator in at
    the genome level. Alternatively, when defined an individual
    mutator may be attached to the trait: this will override the
    default, but must still be compatible with any passed arguments.

    """
    @property
    def name(self):
        return self.__name
    @propery
    def gene_class(self):
        return self.__gene_class
    @property
    def trait_classes(self):
        return self.__trait_classes
    @property
    def traits(self):
        return self.__traits
    @property
    def __init__(self, genetic, gene_category=None, traits=None):
        """
        Initialization method for
        a gene. Method stores the name and the
        genetic object it is binding to, as well as
        other empty parameters.

        :param genetic: a Genetic object, or a subclass of it.
        :param gene_category: a string representing the kind of gene this is. Used for sorting
        :param traits: Any traits this is already known to have.
        """

        #set the basics
        if not isinstance(genetic, Genetic):
            raise GeneticError("Gene was not initialized with Genetic instance")
        self._mother_instance = genetic.get_config()
        self._name = genetic.name
        self._gene_category = gene_category
        if traits is None:
            self.__traits = {}
        else:
            self.__traits = traits
    def get_trait(self, name):
        """
        fetch the trait

        :param name:
        :return:
        """
        return self.__traits[name]
    def set_trait(self, name, trait):
        """
        set the trait

        :param name:
        :param trait:
        :return:
        """
        self.__traits[name] = trait
    def clone(self):
        """

        Clone the gene, returning


        :return:
        """