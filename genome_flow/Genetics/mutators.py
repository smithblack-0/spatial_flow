import tensorflow as tf
from genome_flow.errors import MutatorError


def get(name : str):
    "This function fetches a given mutator"

class mutator:
    """

    Mutators are objects designed to  work on tensorflow variables.
    Their purpose is to, according to some sort of rule, change the
    variable while violating no constraints.


    """