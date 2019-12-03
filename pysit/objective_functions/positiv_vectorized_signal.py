import numpy as np

from pysit.objective_functions.objective_function import ObjectiveFunctionBase
from pysit.util.parallel import ParallelWrapShotNull
from pysit.modeling.pvs_modeling import PVSModeling

__all__ = ['PositivVectorizedSignal']

__docformat__ = "restructuredtext en"


def identity(x):
    return np.asarray(x, dtype=float)

def identity_gradient(x):
    return np.ones_like(np.copy(x), dtype=float)

def get_function(func_id):
    """
    Return function from its id

    Parameter
    ---------
    func_id : str
        function id

    Return
    ------
    Callable object
    """
    switcher = {
        'id': identity
    }

    switcher_gradient = {
        'id': identity_gradient
    }

    return switcher.get(func_id, None), switcher_gradient(func_id, None)


class PositivVectorizedSignal(ObjectiveFunctionBase):

    def __init__(self, solver, sinkhorn_param, tpvs='id', parallel_wrap_shot=ParallelWrapShotNull()):
        self.solver = solver
        self.modeling_tools = PVSModeling(solver)

        self.parallel_wrap_shot = parallel_wrap_shot

        # Name of the function we want to use
        # Dimension of the return type is get dynamically
        # from the returned object.
        self.tpvs = tpvs

        self.T = get_function(tpvs)

    





        