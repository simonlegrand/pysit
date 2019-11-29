

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
from pysit.core.domain import PML
from pysit.util.io import write_data
from pysit.util.io import *


__all__ = ['LayeredGradientMediumModel', 'layered_gradient_medium',
           'Layer', 'set_model_from_file',]


class Layer(object):
    def __init__(self, velocity, thickness, label=None, fixed=False):

        self.velocity = velocity
        self.thickness = thickness
        self.label = label
        self.fixed = fixed

_air   = Layer(300.0, 30, 'air', fixed=True)
_water = Layer(1500.0, 120, 'water', fixed=True)
_rock  = Layer(3200.0, 1800, 'rock')

water_rock = [_water, _rock]

_rock_velocitys = [-288.0, -150.0, -36.0, -18.0, -90.0, 360.0, -60.0, -450.0, 0.0, 78.0]
water_layered_rock = [_water] + [Layer(s+3300, 180, 'rock {0}'.format(i)) for s,i in zip(_rock_velocitys, itertools.count())]


class LayeredGradientMediumModel(GeneratedGalleryModel):

    """ Gallery model for a generic, flat, layered medium. """

    model_name =  "LayeredGradient"

    valid_dimensions = (1,2,3)

    @property #read only
    def dimension(self):
        return self.domain.dim

    supported_physics = ('acoustic',)

    @property
    def z_length(self):
        return round(float(sum([L.thickness for L in self.layers])),5)

    def __init__(self, layers,
                       z_delta=None,
                       min_ppw_at_freq=(6,10.0), # 6ppw at 10hz
                       x_length=None, x_delta=None,
                       y_length=None, y_delta=None,
                       initial_model_style='constant',
                       initial_config={'velocity':1700.0},
                       **kwargs):
        """ Constructor for a constant background model with horizontal reflectors.

        Parameters
        ----------
        layers : list
            List of Layer objects
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
        initial_model_style : {'smooth', 'constant', 'gradient'}
            Setup for the initial model.
        initial_config : dict
            Configuration parameters for initial models.

        Notes
        -----
        * If z_delta is not set, min_ppw_at_freq is used.  z_delta overrides use
        of min_ppw_at_freq.

        * Domain will be covered exactly, so z_delta is the maximum delta, it
        might actually end up being smaller, as the delta is determined by the
        mesh class.

        """

        GeneratedGalleryModel.__init__(self)

        self.layers = layers

        self.min_z_delta = z_delta
        self.min_ppw_at_freq = min_ppw_at_freq

        self.x_length = x_length
        self.x_delta = x_delta
        self.y_length = y_length
        self.y_delta = y_delta

        self.initial_model_style = initial_model_style
        self.initial_config = initial_config

        # Set _domain and _mesh
        self.build_domain_and_mesh(**kwargs)

        # Set _initial_model and _true_model
        self.rebuild_models()

    def build_domain_and_mesh(self, **kwargs):
        """ Constructs a mesh and domain for layered media. """

        # Compute the total depth
        z_length = self.z_length

        x_length = self.x_length
        x_delta = self.x_delta
        y_length = self.y_length
        y_delta = self.y_delta

        # If the minimum z delta is not specified.
        if self.min_z_delta is None: #use min_ppw & peak_frequency
            min_ppw, peak_freq = self.min_ppw_at_freq
            min_velocity = min([L.velocity for L in self.layers])
            wavelength = min_velocity / peak_freq
            z_delta = wavelength / min_ppw
        else:
            z_delta = self.min_z_delta

        z_points = math.ceil((z_length+0.0000001)/z_delta)

        # Set defualt z boundary conditions
        z_lbc = kwargs['z_lbc'] if ('z_lbc' in list(kwargs.keys())) else PML(0.1*z_length, 100.0)
        z_rbc = kwargs['z_rbc'] if ('z_rbc' in list(kwargs.keys())) else PML(0.1*z_length, 100.0)

        domain_configs = list()
        mesh_args = list()

        # If a size of the x direction is specified, determine those parameters
        if x_length is not None:
            if x_delta is None:
                x_delta = z_delta
            x_points = math.ceil(float(x_length+0.00000001)/x_delta)

            # Set defualt x boundary conditions
            x_lbc = kwargs['x_lbc'] if ('x_lbc' in list(kwargs.keys())) else PML(0.1*x_length, 100.0)
            x_rbc = kwargs['x_rbc'] if ('x_rbc' in list(kwargs.keys())) else PML(0.1*x_length, 100.0)

            domain_configs.append((0, x_length, x_lbc, x_rbc))
            mesh_args.append(x_points)

            # the y dimension only exists for 3D proble, so only if x is defined
            if y_length is not None:
                if y_delta is None:
                    y_delta = z_delta
                y_points = math.ceil(float(y_length)/y_delta)

                # Set defualt y boundary conditions
                y_lbc = kwargs['y_lbc'] if ('y_lbc' in list(kwargs.keys())) else PML(0.1*y_length, 100.0)
                y_rbc = kwargs['y_rbc'] if ('y_rbc' in list(kwargs.keys())) else PML(0.1*y_length, 100.0)

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

        sh = self._mesh.shape(as_grid=True)
        _shape_tuple = tuple([1]*(len(sh)-1) + [sh[-1]]) # ones in each dimension except for Z
        _pad_tuple = [(0,n-1) for n in sh]
        _pad_tuple[-1] = (0,0)
        _pad_tuple = tuple(_pad_tuple)

        # Construct true velocity profile
        vp = np.zeros(_shape_tuple)
        grid = self._mesh.mesh_coords(sparse=True)
        ZZ = grid[-1].reshape(_shape_tuple)

        total_filled = 0
        for L in self.layers[::-1]:
            cutoff_depth = self.z_length - total_filled
            vp[ZZ <= cutoff_depth] = L.velocity

            total_filled += L.thickness

        layer_depth = 0
        for L in self.layers[0:-1]:
            layer_depth += L.thickness 
        
        loc = np.where(ZZ >layer_depth)
        vp[loc] = np.linspace(self.layers[1].velocity, self.layers[2].velocity, loc[0].size)

        #######################################################################

        # Construct initial velocity profile:
        if self.initial_model_style == 'constant': # initial_config = {'velocity': 3000.0}
            vp0 = np.ones(_shape_tuple)*self.initial_config['velocity']

        elif self.initial_model_style == 'true': # initial_config = {}, set the initial model as the true model
            vp0 = vp.reshape(-1, )
            vp0.shape = vp.shape

        elif self.initial_model_style == 'smooth': #initial_config = {'sigma':50.0, 'filtersize':8}
            vp0 = blur_image(vp.reshape(-1,),
                             self.initial_config['filtersize'],
                             self.initial_config['sigma'],
                             mesh_deltas=(self._mesh.z.delta,))
            vp0.shape = vp.shape

        elif self.initial_model_style == 'gradient': # initial_config = {'gradient_slope': 1.0}

            gs = self.initial_config['gradient_slope']

            # collect the non-fixed layers for choosing the gradient bounds
            velocities = [L.velocity for L in self.layers if not L.fixed]
            cutoff_depth = 0

            # find the first non-fixed layer to start the gradient at.
            for L in self.layers:
                if L.fixed:
                    cutoff_depth += L.thickness
                else:
                    break
            vp0 = vp.copy()
            loc = np.where(ZZ > cutoff_depth)
            vp0[loc] = np.linspace(min(velocities), gs*np.max(velocities), loc[0].size)

        elif self.initial_model_style == 'layer':
            vels_init = self.initial_config['initial_velocity']
            thick_init = self.initial_config['initial_thickness']
    
            layer_init = [Layer(s, t, 'Layer_init_ {0}'.format(i)) for s,t,i in zip(vels_init, thick_init, itertools.count())]

            vp0 = np.zeros(_shape_tuple)
            grid = self._mesh.mesh_coords(sparse=True)
            ZZ = grid[-1].reshape(_shape_tuple)

            total_filled = 0
            for L in layer_init[::-1]:
                cutoff_depth = self.z_length - total_filled
                vp0[ZZ <= cutoff_depth] = L.velocity
                total_filled += L.thickness

        # Fix the fixed layers
        old_depth = 0
        for L in self.layers:
            depth = old_depth + L.thickness
            if L.fixed:
                vp0[(ZZ >= old_depth) & (ZZ < depth)] = L.velocity
            old_depth = depth


        # Construct final padded velocity profiles
        C = np.pad(vp, _pad_tuple, 'edge').reshape(self._mesh.shape())
        C0 = np.pad(vp0, _pad_tuple, 'edge').reshape(self._mesh.shape())
        self._true_model = C
        self._initial_model = C0

def layered_medium(layers=water_layered_rock, **kwargs):
    """ Friendly wrapper for instantiating the layered medium model. """

    # Setup the defaults
    model_config = dict(z_delta=None,
                        min_ppw_at_freq=(6,10.0), # 6ppw at 10hz
                        x_length=None, x_delta=None,
                        y_length=None, y_delta=None,
                        initial_model_style='smooth',
                        initial_config={'sigma':100.0, 'filtersize':100})

    # Make any changes
    model_config.update(kwargs)

    return LayeredMediumModel(layers, **model_config).get_setup()


def layered_gradient_medium(model_param=None, initial_model_style=None, initial_config=None, 
                        dx = 0.01, dz = 0.01, water_layer_depth = 0.05, TrueModelFileName=None, 
                        InitialModelFileName=None, **kwargs):

    x_length = model_param['x_length']
    z_depth = model_param['z_depth']
    layer_thickness = model_param['layer_thickness']
    vels = model_param['velocity']

    nbx = 10
    nbz = 10 
    pml_width = [0.5,0.5]

    nx = int(x_length/dx) + 1
    nz = int(z_depth/dz) + 1

    nl = np.shape(layer_thickness)[0]

    n_layer = np.zeros(nl)
    for i in range(nl):
        n_layer[i] = int(layer_thickness[i]/dz)

    n_layer[0] = n_layer[0]  # + nbz
    n_layer[nl-1]= n_layer[nl-1] # + 1  # + nbz

    nxt = nx # + 2*nbx
    nzt = nz # + 2*nbz

    Layerall = list()
    for i in range(nl):
        Layers = Layer(vels[i], n_layer[i]*dz, fixed=False)
        Layerall += [Layers]

    x_lbc = kwargs['x_lbc'] if ('x_lbc' in kwargs) else PML(0.1, 100)
    x_rbc = kwargs['x_rbc'] if ('x_rbc' in kwargs) else PML(0.1, 100)
    z_lbc = kwargs['z_lbc'] if ('z_lbc' in kwargs) else PML(0.1, 100)
    z_rbc = kwargs['z_rbc'] if ('z_rbc' in kwargs) else PML(0.1, 100)

    kwargs['x_lbc'] = PML(pml_width[0], 100)
    kwargs['x_rbc'] = PML(pml_width[0], 100)
    kwargs['z_lbc'] = PML(pml_width[1], 100)
    kwargs['z_rbc'] = PML(pml_width[1], 100)

    model_config = dict(z_delta=dz,
                        x_length=dx*(nxt-1), x_delta=dx,
                        initial_model_style=initial_model_style,
                        initial_config=initial_config, **kwargs)

    C, C0, m, d = LayeredGradientMediumModel(Layerall, **model_config).get_setup()

    if TrueModelFileName is not None:
        ot = (0.0,0.0)
        dt = (dz, dx)
        nt = m._shapes[(False, True)]
        B  = C.reshape(nt).transpose()
        nt = (nt[1], nt[0])
        write_data(TrueModelFileName, B, ot, dt, nt)

    if InitialModelFileName is not None:
        ot = (0.0,0.0)
        dt = (dz, dx)
        nt = m._shapes[(False, True)]
        B  = C0.reshape(nt).transpose()
        nt = (nt[1], nt[0])
        write_data(InitialModelFileName, B, ot, dt, nt)


    return C, C0, m, d


def set_model_from_file(Modelfile, 
                        initial_config={'sigma': 1.0, 'filtersize': 8},
                        initial_model_style='smooth',
                        **kwargs):

    """
        function to set up the model from the Modelfile

        Input:
        Modelfile: The name of the model file. The file should following struction
                   A.data - the data of the model 
                   A.o    - the origions of each dimension
                   A.d    - the delta of each dimension
                   A.n    - the size of each dimension
        
        Optional Input:
        If you want to create a smooth model from the given velocity model, you can 
        use the following inputs:
        initial_config: 'sigma' and 'filtersize' define the level of the smoothness
        initial_model_style: default is 'smooth', you can also select 'gradient' then 
                             you will get a linear velocity model

        Key word arguments:
        You can define the PML as follows:
        kwargs['x_lbc'] = PML(0.1, 100)
        kwargs['x_rbc'] = PML(0.1, 100)
        kwargs['z_lbc'] = PML(0.1, 100)
        kwargs['z_rbc'] = PML(0.1, 100)
    """

    [vels, ot, dt, nt] = read_data(Modelfile)

    C, C0, m, d = three_layered_medium(dx=dt[1], dz=dt[0],
                                       nx=nt[1], nz=nt[0], 
                                       initial_model_style=initial_model_style,
                                       initial_config=initial_config,
                                       TrueModelFileName=None, InitialModelFileName=None,
                                       **kwargs)

    C = vels.transpose().reshape(C.shape)
    # C0 = copy.deepcopy(C)

    return C, m, d

    
    

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # ASD = LayeredMediumModel(water_layered_rock)
    # ASD = LayeredMediumModel(water_layered_rock, initial_model_style='smooth', initial_config={'sigma':100, 'filtersize':150})
    # ASD = LayeredMediumModel(water_layered_rock, initial_model_style='gradient')
    # ASD = LayeredMediumModel(water_layered_rock, initial_model_style='constant', initial_config={'velocity':3000})
    # SD = LayeredMediumModel(water_layered_rock, x_length=2000.0, y_length=1000.0)

    C, C0, m, d = layered_medium(x_length=2000)

    fig = plt.figure()
    fig.add_subplot(2,1,1)
    vis.plot(C, m)
    fig.add_subplot(2,1,2)
    vis.plot(C0, m)
    plt.show()

    C, C0, m, d = three_layered_medium(TrueModelFileName='testtrue.mat',InitialModelFileName='testInitial.mat',
                                       initial_model_style='smooth',
                                       initial_config={'sigma': 2.0, 'filtersize': 16})
    C, m, d = set_model_from_file('testtrue.mat')
    C0, m, d = set_model_from_file('testInitial.mat')
    fig = plt.figure()
    fig.add_subplot(2, 1, 1)
    vis.plot(C, m)
    fig.add_subplot(2, 1, 2)
    vis.plot(C0, m)
    plt.show()

#   print(np.max(C-C0))

  

  
