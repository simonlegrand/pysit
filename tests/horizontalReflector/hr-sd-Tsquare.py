# Std import block
import time
import pickle
import numpy as np

import copy
import sys
import scipy.io as sio

from pysit import *
from pysit.gallery import horizontal_reflector
from pysit.util.io import *
from pysit.util.parallel import *

from mpi4py import MPI

if __name__ == '__main__':
    # Setup
    ExpDir = '.'

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    pwrap = ParallelWrapShot()

    if rank == 0:
        ttt = time.time()

    #   Define Domain
    pmlx = PML(0.1, 100)
    pmlz = PML(0.1, 100)

    x_config = (0.1, 1.0, pmlx, pmlx)
    z_config = (0.1, 0.8, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)
    m = CartesianMesh(d, 91, 71)

    if rank == 0:
        sys.stdout.write('Horizontal reflector medium \n')

    #   Generate true wave speed
    C, C0, m, d = horizontal_reflector(m)

    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = zmin + (1./9.)*zmax

    Nshots = size
    Nreceivers = 'max'
    Ric_freq = 15.0
    sys.stdout.write("{0}: {1}\n".format(rank, Nshots / size))

    shots = equispaced_acquisition(m,
                                   RickerWavelet(Ric_freq),
                                   sources=Nshots,
                                   source_depth=zpos,
                                   source_kwargs={},
                                   receivers=Nreceivers,
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   parallel_shot_wrap=pwrap,
                                   )

    # shots_freq = copy.deepcopy(shots)

    # Define and configure the wave solver
    t_range = (0.0,2.5)

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=6,
                                         trange=t_range,
                                         kernel_implementation='cpp')

    # Generate synthetic Seismic data
    if rank == 0:
        sys.stdout.write('Model parameters setting: \n')
        sys.stdout.write('Nshots = %d \n' %Nshots)
        if Nreceivers == 'max':
            sys.stdout.write('Nreceivers = %d \n' %m.x.n)
        else:
            sys.stdout.write('Nreceivers = %d \n' %Nreceivers)
        sys.stdout.write('Ricker wavelet frequency = %.1f Hz \n' %Ric_freq)
        sys.stdout.write('Recording time = %.1f s\n' %t_range[1])
        sys.stdout.write('Generating data... \n')

    initial_model = solver.ModelParameters(m,{'C': C0})
    generate_seismic_data(shots, solver, initial_model)
    wavefield_initial = comm.gather(shots[0].receivers.data, root=0)

    # Generate synthetic Seismic data
    base_model = solver.ModelParameters(m,{'C': C})
    tt = time.time()
    generate_seismic_data(shots, solver, base_model)
    wavefield_true = comm.gather(shots[0].receivers.data, root=0)

    sys.stdout.write('{1}:Data generation: {0}s\n'.format(time.time()-tt,rank))
    sys.stdout.flush()

    comm.Barrier()

    if rank == 0:
        tttt = time.time()-ttt
        sys.stdout.write('Total wall time: {0}\n'.format(tttt))
        sys.stdout.write('Total wall time/shot: {0}\n'.format(tttt/Nshots))

    ############# Set up objective function ##############
    ot_param = { 'sinkhorn_iterations'          : 10000,
                 'sinkhorn_tolerance'           : 1.0e-9,
                 'epsilon_maxsmooth'            : 1.0e-5,   # for the smoothing of the max(., 0)
                 'successive_over_relaxation'   : 1.4,
                 'trans_func_type'              : 'square',  ## smooth_max ## exp ## square ## id ##
                 'epsilon_kl'                   : 1e-2,
                 'lamb_kl'                      : 1.0,
                 't_scale'                      : 1.0,
                 'x_scale'                      : 1.0,
                 'nt_resampling'                : 128,
                 'sinkhorn_initialization'      : True,
                #  'Noise'                        : False,
                 'N_receivers'                  : Nreceivers,
                 'filter_op'                    : False,
                 'freq_band'                    : [1, 30.0],
               }

    #### Least-squares objective function
    # if rank == 0:
    #     print('Least-squares...')
    # objective = TemporalLeastSquares(solver, parallel_wrap_shot=pwrap)

    #### Sinkhorn-Divergence objective function
    if rank == 0:
        print('Sinkhorn Divergence...')
        print('Sinkhorn Divergence parameters setting:')
        print('trans_func_type = %s' %ot_param['trans_func_type'])
        print('sinkhorn_initialization = %s' %ot_param['sinkhorn_initialization'])
        print('sinkhorn_epsilon_kl = %.1f' %ot_param['epsilon_kl'])
        print('sinkhorn_lamb_kl = %.1f' %ot_param['lamb_kl'])
        print('sinkhorn_t_scale = %.1f' %ot_param['t_scale'])
        print('sinkhorn_x_scale = %.1f' %ot_param['x_scale'])
        print('sinkhorn_nt_resampling = %d' %ot_param['nt_resampling'])

    objective = SinkhornDivergence(solver, ot_param=ot_param, parallel_wrap_shot=pwrap)

    # Define the inversion algorithm   
    line_search = 'backtrack'
    # line_search = ('constant', 1e-16)
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

    # print('Running GradientDescent...')
    # invalg = GradientDescent(objective)

    # print('Running PQN...')
    # bound = [1.5, 6.5]
    # Proj_Op1 = BoxConstraintPrj(bound)
    # invalg_1 = PQN(objective, proj_op=Proj_Op1, memory_length=10)

    if rank == 0:
        print('Running LBFGS...')

    invalg = LBFGS(objective, memory_length=10)
    initial_value = solver.ModelParameters(m,{'C': C0})

    # Execute inversion algorithm
    tt = time.time()

    nsteps = 20
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
    wavefield_inverted = comm.gather(shots[0].receivers.data, root=0)

    # SaveData
    if rank == 0:
        ############ Saving results ###########################
        with open('mesh.p', 'wb') as f:
            pickle.dump(m, f)

        conv_vals = np.array([v for k,v in list(invalg.objective_history.items())])

        initial_value.data = invalg.gradient_history[0].data
        gradient = initial_value.without_padding().data.reshape(m.shape(as_grid=True)).transpose()

        ns = int(np.shape(wavefield_true)[0]/2)

        output = {'conv': conv_vals,
                'inverted': C_inverted,
                'true': C.reshape(m.shape(as_grid=True)).transpose(),
                'initial': C0.reshape(m.shape(as_grid=True)).transpose(),
                'wavefield_true': wavefield_true[ns],
                'wavefield_initial': wavefield_initial[ns],
                'wavefield_inverted': wavefield_inverted[ns],
                'gradient': gradient,
                'x_range': [d.x.lbound, d.x.rbound],
                'z_range': [d.z.lbound, d.z.rbound],
                't_range': t_range,
                'obj_name': objective.name(),
                }

        sio.savemat('./output.mat', output)
