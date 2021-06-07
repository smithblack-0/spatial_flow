class trait:
    """

    A trait catagorizes and formalizes the particulars of
    what can be mutated, how a copy of the object of interest
    may be made, and other important concepts.

    This is the base class

    """

    def copy(self):
        pass

    def mutate(self):
        pass


class Variable(trait):
    """

    The variable trait

    A variable trait is an implementation
    of a tensorflow variable and associated
    tracking functionality

    """



class Genetic: