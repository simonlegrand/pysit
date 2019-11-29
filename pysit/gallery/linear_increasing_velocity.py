import copy
import tarfile
import os
import os.path
import itertools
import math

import numpy as np
import scipy.signal as signal

from pysit.util.image_processing import blur_image
from pysit.gallery.gallery_base import GeneratedGalleryModel

from pysit import * #PML, Domain


__all__ = ['LinearIncreasedMediumModel', 'linear_increasing_velocity']


class LinearIncreasedMediumModel(GeneratedGalleryModel):

    """ Gallery model for a linear increasing velocity medium. """

    model_name =  "LinearIncreased"

    valid_dimensions = (1,2,3)

    @property #read only
    def dimension(self):
        return self.domain.dim

    supported_physics = ('acoustic',)

    @property
    def z_length(self):
        return float(self.thickness)

    def __init__(self, model_param_set=None,
                       min_ppw_at_freq=(6,10.0), # 6ppw at 10hz
                       y_length=None, y_delta=None, z_delta=None,
                       **kwargs):
        """ Constructor for a constant background model with horizontal reflectors.

        Parameters
        ----------
        z_delta : float, optional
            Minimum mesh spacing in depth direction, see Notes.
        min_ppw_at_freq : tuple (int, float)
            Tuple with structure (min_ppw, peak_freq) to set the minimum points-per-wavelength at the given peak frequency.
        x_length : float
            Physical size in x direction
        x_delta : float
            Grid spacing in x direction
        y_length : float
            Physical size in y direction
        y_delta : float
            Grid spacing in y direction

        Notes
        -----
        * If z_delta is not set, min_ppw_at_freq is used.  z_delta overrides use
        of min_ppw_at_freq.

        * Domain will be covered exactly, so z_delta is the maximum delta, it
        might actually end up being smaller, as the delta is determined by the
        mesh class.

        """

        GeneratedGalleryModel.__init__(self)

        self.thickness = model_param_set['z_depth']

        self.min_z_delta = z_delta
        self.min_ppw_at_freq = min_ppw_at_freq

        self.x_length = model_param_set['x_length']
        self.x_delta = model_param_set['x_delta']
        self.y_length = y_length
        self.y_delta = y_delta

        self.vp_initial = model_param_set['vp_initial']
        self.vp_true = model_param_set['vp_true']
        self.alpha_initial = model_param_set['alpha_initial']
        self.alpha_true = model_param_set['alpha_true']

        # Set _domain and _mesh
        self.build_domain_and_mesh(**kwargs)

        # Set _initial_model and _true_model
        self.rebuild_models()

    def build_domain_and_mesh(self, **kwargs):
        """ Constructs a mesh and domain for linear-increased-velocity media. """

        # Compute the total depth
        z_length = self.z_length

        x_length = self.x_length
        x_delta = self.x_delta
        y_length = self.y_length
        y_delta = self.y_delta

        # If the minimum z delta is not specified.
        if self.min_z_delta is None: #use min_ppw & peak_frequency
            min_ppw, peak_freq = self.min_ppw_at_freq
            min_velocity = self.vp_true
            wavelength = min_velocity / peak_freq
            z_delta = wavelength / min_ppw
        else:
            z_delta = self.min_z_delta

        z_points = math.ceil(z_length/z_delta)
        # Set defualt z boundary conditions
        z_lbc = kwargs['z_lbc'] if ('z_lbc' in kwargs.keys()) else PML(0.1*z_length, 100.0)
        z_rbc = kwargs['z_rbc'] if ('z_rbc' in kwargs.keys()) else PML(0.1*z_length, 100.0)

        domain_configs = list()
        mesh_args = list()

        # If a size of the x direction is specified, determine those parameters
        if x_length is not None:
            if x_delta is None:
                x_delta = z_delta
            x_points = math.ceil(float(x_length)/x_delta)
            # Set defualt x boundary conditions
            x_lbc = kwargs['x_lbc'] if ('x_lbc' in kwargs.keys()) else PML(0.1*x_length, 100.0)
            x_rbc = kwargs['x_rbc'] if ('x_rbc' in kwargs.keys()) else PML(0.1*x_length, 100.0)

            domain_configs.append((0, x_length, x_lbc, x_rbc))
            mesh_args.append(x_points)

            # the y dimension only exists for 3D proble, so only if x is defined
            if y_length is not None:
                if y_delta is None:
                    y_delta = z_delta
                y_points = math.ceil(float(y_length)/y_delta)

                # Set defualt y boundary conditions
                y_lbc = kwargs['y_lbc'] if ('y_lbc' in kwargs.keys()) else PML(0.1*y_length, 100.0)
                y_rbc = kwargs['y_rbc'] if ('y_rbc' in kwargs.keys()) else PML(0.1*y_length, 100.0)

                domain_configs.append((0, y_length, y_lbc, y_rbc))
                mesh_args.append(y_points)

        domain_configs.append((0, z_length, z_lbc, z_rbc))
        mesh_args.append(z_points)

        self._domain = RectangularDomain(*domain_configs)

        # Build mesh
        mesh_args = [self._domain] + mesh_args
        self._mesh = CartesianMesh(*mesh_args)

    def rebuild_models(self):
        """ Rebuild the true and initial models based on the current configuration."""

        vp_initial = self.vp_initial
        vp_true = self.vp_true
        alpha_initial = self.alpha_initial
        alpha_true = self.alpha_true

        sh = self._mesh.shape(as_grid=True)
        _shape_tuple = tuple([1]*(len(sh)-1) + [sh[-1]]) # ones in each dimension except for Z
        _pad_tuple = [(0,n-1) for n in sh]
        _pad_tuple[-1] = (0,0)
        _pad_tuple = tuple(_pad_tuple)

        # Construct true velocity profile
        vp = np.zeros(_shape_tuple)
        grid = self._mesh.mesh_coords(sparse=True)
        ZZ = grid[-1].reshape(_shape_tuple)
        vp = vp_true + alpha_true * ZZ

        # Construct initial velocity profile:
        vp0 = np.zeros(_shape_tuple)
        grid = self._mesh.mesh_coords(sparse=True)
        ZZ = grid[-1].reshape(_shape_tuple)
        vp0 = vp_initial + alpha_initial * ZZ

        # Construct final padded velocity profiles
        C = np.pad(vp, _pad_tuple, 'edge').reshape(self._mesh.shape())
        C0 = np.pad(vp0, _pad_tuple, 'edge').reshape(self._mesh.shape())
        self._true_model = C
        self._initial_model = C0

def linear_increasing_velocity(model_param_set, **kwargs):
    """ Friendly wrapper for instantiating the linear-increased-velocity medium model. """

    # Setup the defaults
    model_config = dict(z_delta=None,
                        min_ppw_at_freq=(6,10.0), # 6ppw at 10hz
                        y_length=None, y_delta=None)

    # Make any changes
    model_config.update(kwargs)

    return LinearIncreasedMediumModel(model_param_set, **model_config).get_setup()
