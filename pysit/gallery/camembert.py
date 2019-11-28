

import numpy as np

from pysit.gallery.gallery_base import GeneratedGalleryModel
from pysit.util.io import write_data
from pysit.util.io import *

__all__ = ['CamembertModel', 'camembert']


class CamembertModel(GeneratedGalleryModel):

    """ Gallery model for constant background plus simple horizontal reflectors. """

    model_name = "Horizontal Reflector"

    valid_dimensions = (1,2,3)

    @property
    def dimension(self):
        return self.domain.dim

    supported_physics = ('acoustic',)

    def __init__(self, mesh,
                       camembert_radius=0.25, # as percentage of domain
                       camembert_velocity=2.5,
                       background_velocity=2.0,
                       ):
        """ Constructor for a constant background model with horizontal reflectors.

        Parameters
        ----------
        mesh : pysit mesh
            Computational mesh on which to construct the model
        camembert_radius : the radius of the camembert, float
        camembert_velocity : the velocity inside the camembert, float
        background_velocity : float


        """

        GeneratedGalleryModel.__init__(self)


        self.camembert_radius = camembert_radius
        self.camembert_velocity = camembert_velocity

        self.background_velocity = background_velocity

        self._mesh = mesh
        self._domain = mesh.domain
        # Set _initial_model and _true_model
        self.rebuild_models()

    def rebuild_models(self, camembert_radius=None, camembert_velocity=None, background_velocity=None):
        """ Rebuild the true and initial models based on the current configuration."""

        if camembert_radius is not None:
            self.camembert_radius = camembert_radius

        if camembert_velocity is not None:
            self.camembert_velocity = camembert_velocity

        if background_velocity is not None:
            self.background_velocity = background_velocity

        C0 = self.background_velocity*np.ones(self._mesh.shape())

        dC = self._build_camembert()

        self._initial_model = C0
        self._true_model = C0 + dC

    def _build_camembert(self):

        mesh = self.mesh
        domain = self.domain

        grid = mesh.mesh_coords()

        ndim = len(grid)

        cent_point = []
        dC = np.zeros(mesh.shape())
        for i in range(ndim):
            cent_point.append((grid[i][0] + grid[i][-1]) / 2.0)

        for i in range(len(grid[0])):
            a = 0.0
            for j in range(ndim):
                a += (grid[j][i] - cent_point[j])**2.0
            a = np.sqrt(a)
            if a < self.camembert_radius:
                dC[i] = self.camembert_velocity - self.background_velocity 


        # # can set any defaults here
        # if self.pulse_style == 'gaussian_derivative':
        #     pulse_config = {}
        # elif self.pulse_style == 'gaussian':
        #     pulse_config = {}

        # # update to any user defined defaults
        # pulse_config.update(self.pulse_config)

        # for d,s in zip(self.reflector_depth, self.reflector_scaling):

        #     # depth is a percentage of the length
        #     depth  = domain.z.lbound + d * domain.z.length

        #     pulse = _pulse_functions[self.pulse_style](ZZ-depth, self.drop_threshold, **pulse_config)
        #     dC += s*pulse

        return dC

def camembert( mesh, **kwargs):
    """ Friendly wrapper for instantiating the horizontal reflector model. """

    # Setup the defaults
    model_config = dict(camembert_radius=0.1,  # as percentage of domain
                        camembert_velocity=2.5,
                        background_velocity=2.0,
                        )

    # Make any changes
    model_config.update(kwargs)

    return CamembertModel(mesh, **model_config).get_setup()

if __name__ == '__main__':

  from pysit import *

  #       Define Domain
  pmlx = PML(0.1, 100)
  pmlz = PML(0.1, 100)

  x_config = (0.1, 1.0, pmlx, pmlx)
  z_config = (0.1, 0.8, pmlz, pmlz)

  d = RectangularDomain(x_config, z_config)

  m = CartesianMesh(d, 91, 71)

  #       Generate true wave speed
  C, C0, m, d = camembert(m)

  import matplotlib.pyplot as plt

  fig = plt.figure()
  fig.add_subplot(2,1,1)
  vis.plot(C, m)
  fig.add_subplot(2,1,2)
  vis.plot(C0, m)
  plt.show()
