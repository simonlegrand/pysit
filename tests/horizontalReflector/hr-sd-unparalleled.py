# Std import block
import time
import numpy as np
import scipy.io as sio

import pickle

from pysit import *
from pysit.gallery import horizontal_reflector

if __name__ == '__main__':
    # Setup

    #   Define Domain
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)

    m = CartesianMesh(d, 91, 71)

    #   Generate true wave speed
    C, C0, m, d = horizontal_reflector(m)


    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = zmin + (1./9.)*zmax
    Nreceivers = 60
    shots = equispaced_acquisition(m,
                                   RickerWavelet(10.0),
                                   sources=1,
                                   source_depth=zpos,
                                   source_kwargs={},
                                   receivers=Nreceivers,
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   )

    # Define and configure the wave solver
    t_range = (0.0,1.5)

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=6,
                                         trange=t_range,
                                         kernel_implementation='cpp')

    initial_model = solver.ModelParameters(m,{'C': C0})
    generate_seismic_data(shots, solver, initial_model)
    wavefield_initial = shots[0].receivers.data

    # Generate synthetic Seismic data
    tt = time.time()
    wavefields =  []
    base_model = solver.ModelParameters(m,{'C': C})
    generate_seismic_data(shots, solver, base_model, wavefields=wavefields)
    wavefield_true = shots[0].receivers.data

    print('Data generation: {0}s'.format(time.time()-tt))

    ############# Set up objective function ##############

    #### Least-squares objective function
    # print('Least-squares...')
    # objective = TemporalLeastSquares(solver)

    #### Sinkhorn-Divergence objective function
    print('Sinkhorn Divergence...')
    ot_param = { 'sinkhorn_iterations'          : 10000,
                 'sinkhorn_tolerance'           : 1.0e-9,
                 'epsilon_maxsmooth'            : 1.0e-5,   # for the smoothing of the max(., 0)
                 'successive_over_relaxation'   : 1.4,
                 'trans_func_type'              : 'smooth_max',  ## smooth_max ## exp ##
                 'epsilon_kl'                   : 1e-2,
                 'lamb_kl'                      : 1.0,
                 't_scale'                      : 10.0,
                 'x_scale'                      : 10.0,
                 'nt_resampling'                : 128,
                 'sinkhorn_initialization'      : True,
                 'N_receivers'                  : Nreceivers,
                 'filter_op'                    : False,
                 'freq_band'                    : [1, 30.0],
               }
    objective = SinkhornDivergence(solver, ot_param=ot_param)

    # Define the inversion algorithm
    invalg = LBFGS(objective)
    initial_value = solver.ModelParameters(m,{'C': C0})

    # Execute inversion algorithm
    print('Running LBFGS...')
    tt = time.time()

    nsteps = 2

    status_configuration = {'value_frequency'           : 1,
                            'residual_frequency'        : 1,
                            'residual_length_frequency' : 1,
                            'objective_frequency'       : 1,
                            'step_frequency'            : 1,
                            'step_length_frequency'     : 1,
                            'gradient_frequency'        : 1,
                            'gradient_length_frequency' : 1,
                            'run_time_frequency'        : 1,
                            'alpha_frequency'           : 1,
                            }

    # line_search = ('constant', 1e-16)
    line_search = 'backtrack'

    result = invalg(shots, initial_value, nsteps,
                    line_search=line_search,
                    status_configuration=status_configuration, verbose=True, write=True)

    print('...run time:  {0}s'.format(time.time()-tt))

    initial_value.data = result.C
    C_cut = initial_value.without_padding().data
    C_inverted = C_cut.reshape(m.shape(as_grid=True)).transpose()

    ####################################################################################################
    # Save wavefield
    inverted_model = solver.ModelParameters(m,{'C': C_cut})
    generate_seismic_data(shots, solver, inverted_model)
    wavefield_inverted = shots[0].receivers.data

    ############ Saving results ###########################
    with open('mesh.p', 'wb') as f:
        pickle.dump(m, f)

    conv_vals = np.array([v for k,v in list(invalg.objective_history.items())])

    initial_value.data = invalg.gradient_history[0].data
    gradient = initial_value.without_padding().data.reshape(m.shape(as_grid=True)).transpose()

    # ns = int(np.shape(wavefield_true)[0]/2)

    output = {'conv': conv_vals,
              'inverted': C_inverted,
              'true': C.reshape(m.shape(as_grid=True)).transpose(),
              'initial': C0.reshape(m.shape(as_grid=True)).transpose(),
              'wavefield_true': wavefield_true,
              'wavefield_initial': wavefield_initial,
              'wavefield_inverted': wavefield_inverted,
              'gradient': gradient,
              'x_range': [d.x.lbound, d.x.rbound],
              'z_range': [d.z.lbound, d.z.rbound],
              't_range': t_range,
              'obj_name': objective.name(),
              }

    sio.savemat('./output.mat', output)