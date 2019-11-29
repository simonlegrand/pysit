import numpy as np

from pysit.objective_functions.objective_function import ObjectiveFunctionBase
from pysit.util.parallel import ParallelWrapShotNull
from pysit.modeling.pvs_modeling import PVSModeling

__all__ = ['PositivVectorizedSignal']

__docformat__ = "restructuredtext en"

class PositivVectorizedSignal(ObjectiveFunctionBase):

    def __init__(self, solver, parallel_wrap_shot=ParallelWrapShotNull(), tpvs='exp', ipvs=1):
        self.solver = solver
        self.modeling_tools = PVSModeling(solver)

        self.parallel_wrap_shot = parallel_wrap_shot