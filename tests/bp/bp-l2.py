# Std import block
import time
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal

import copy
from shutil import copy2
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
import scipy.io as sio

from pysit import *
from pysit.gallery import bp
from pysit.util.io import *
from pysit.util.compute_tools import *
from pysit.util.parallel import *

from mpi4py import MPI

if __name__ == '__main__':
    # Setup
    RootDir = '/scratch/miyu/results-pysit/'
    SubDir = ''

#    ExpDir = RootDir + SubDir
    ExpDir = '.'

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    pwrap = ParallelWrapShot()

    if rank == 0:
        ttt = time.time()

    # Set up domain, mesh and velocity model
    #C, C0, m, d = bp(pixel_scale='mini', initial_model_style='smooth_width', initial_config={'sigma':1000.})
    #C, C0, m, d = bp(patch='left', pixel_scale='mini', initial_model_style='gradient', initial_config={'min':1500., 'max':4000.})
    #C, C0, m, d = bp(patch='right', pixel_scale='mini', initial_model_style='constant', initial_config={'velocity':3000.})
    C, C0, m, d = bp(patch='right', pixel_scale='mini', initial_model_style='smooth_low_pass', initial_config={'freq':1/3000.})

    m_shape = m._shapes[(False,True)]
    pmlx = PML(0.5, 1000)
    pmlz = PML(0.5, 1000)

    x_config = (d.x.lbound/1000.0, d.x.rbound/1000.0, pmlx, pmlx)
    z_config = (d.z.lbound/1000.0, d.z.rbound/1000.0, pmlz, pmlz)

    d = RectangularDomain(x_config, z_config)
    m = CartesianMesh(d, m_shape[0], m_shape[1])

    C = C/1000
    C0 = C0/1000

    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = 0.05 * 5.0

    Nshots = size
    Nreceivers = 301
    sys.stdout.write("{0}: {1}\n".format(rank, Nshots / size))

    shots = equispaced_acquisition(m,
                                   RickerWavelet(5.0),
                                   sources=Nshots,
                                   source_depth=zpos,
                                   source_kwargs={},
                                   receivers=Nreceivers,
                                   receiver_depth=zpos,
                                   receiver_kwargs={},
                                   parallel_shot_wrap=pwrap,
                                   )

    shots_freq = copy.deepcopy(shots)

    # Define and configure the wave solver
    trange = (0.0,7.0)

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=2,
                                         trange=trange,
                                         kernel_implementation='cpp',
                                         max_C=5.0) # The dt is automatically fixed for given max_C (velocity)

    # Generate synthetic Seismic data
    sys.stdout.write('Generating data...')

    initial_model = solver.ModelParameters(m,{'C': C0})
    generate_seismic_data(shots, solver, initial_model)
    wavefield_initial = comm.gather(shots[0].receivers.data, root=0)

    base_model = solver.ModelParameters(m,{'C': C})
    tt = time.time()
    generate_seismic_data(shots, solver, base_model)
    wavefield_true = comm.gather(shots[0].receivers.data, root=0)

    if rank==0:
        ## Save initial and true wavefield
        tmp_save_wavefield = {'wavefield_true': wavefield_true,
                              'wavefield_initial': wavefield_initial,
                             }
        sio.savemat(ExpDir + '/wavefields.mat', tmp_save_wavefield)

        ## Save initial and true model
        nt = m.shape(as_grid=True)
        nt = (nt[1], nt[0])
        dt = (m.parameters[1].delta, m.parameters[0].delta)
        ot = (d.z.lbound, d.x.lbound)

        write_data(ExpDir + '/model-true.mat', C.reshape(m.shape(as_grid=True)).transpose(), ot, dt, nt)
        write_data(ExpDir + '/model-initial.mat', C0.reshape(m.shape(as_grid=True)).transpose(), ot, dt, nt)
    ###################################

    sys.stdout.write('{1}:Data generation: {0}s\n'.format(time.time()-tt,rank))
    sys.stdout.flush()

    comm.Barrier()

    if rank == 0:
        tttt = time.time()-ttt
        sys.stdout.write('Total wall time: {0}\n'.format(tttt))
        sys.stdout.write('Total wall time/shot: {0}\n'.format(tttt/Nshots))

    # Filter data
    n_timesmp = shots[0].receivers.data.shape[0]
    T_max = solver.tf
    freq_band =[3.0, 9.0]
    filter_op1 = band_pass_filter(n_timesmp, T_max, freq_band, transit_freq_length=0.5, padding_zeros=True, nl=500, nr=500)
    
    # Set up objective function

    # Least-squares objective function
    print('Least-squares...')
    objective = TemporalLeastSquares(solver, filter_op=filter_op1, imaging_period=1, parallel_wrap_shot=pwrap)

    # Envelope objective function
    # print('Envelope...')
    # objective = TemporalEnvelope(solver, envelope_power=2.0, filter_op=filter_op1, imaging_period=1, parallel_wrap_shot=pwrap)

    # Cross-correlation objective function
    # print('Cross-correlation...')
    # objective = TemporalCorrelate(solver, filter_op=filter_op1, imaging_period=1, parallel_wrap_shot=pwrap)   

    # Optimal transportation objective function with linear transformation
    # print('Ot with linear transformation...')
    # objective = TemporalOptimalTransport(solver, filter_op=filter_op1, imaging_period=1, transform_mode='linear', c_ratio=2.0, parallel_wrap_shot=pwrap)

    # Optimal transportation objective function with quadratic transformation
    # print('Ot with quadratic transformation...')
    # objective = TemporalOptimalTransport(solver, filter_op=filter_op1, imaging_period=1, transform_mode='quadratic', parallel_wrap_shot=pwrap)

    # Optimal transportation objective function with absolute transformation
    # print('Ot with absolute transformation...')
    # objective = TemporalOptimalTransport(solver, filter_op=filter_op1, imaging_period=1, transform_mode='absolute', parallel_wrap_shot=pwrap)

    # Optimal transportation objective function with exponential transformation
    # print('Ot with exponential transformation...')
    # objective = TemporalOptimalTransport(solver, filter_op=filter_op1, imaging_period=1, transform_mode='exponential', exp_a=1.0, parallel_wrap_shot=pwrap)  

    # Optimal transportation Maximum Mean Discrepancies objective function
    ot_param = { 'sinkhorn_iterations'          : 10000,
                 'sinkhorn_tolerance'           : 1.0e-9,
                 'epsilon_maxsmooth'            : 1.0e-5,   # for the smoothing of the max(., 0)
                 'successive_over_relaxation'   : 1.4,
                 'sign_option'                  : "pos+neg",
                 'epsilon_kl'                   : 1e-1,
                 'lamb_kl'                      : 1.0,
                 't_range'                      : 10.0,
                 'x_range'                      : 10.0,
                 'nt_resampling'                : 128,
               }
    # print('Otmmd...')
    # objective = OptimalTransportMaximumMeanDiscrepancies(solver, ot_param=ot_param, filter_op=filter_op1, parallel_wrap_shot=pwrap)  
    
    # Define the inversion algorithm
    line_search = 'backtrack'
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

    print('Running LBFGS...')
    invalg = LBFGS(objective, memory_length=10)

    ################ Use otmmd results as the starting model ##############
    #print('Loading otmmd results as the starting model......')
    #C_ot_load = loadmat(RootDir + '/marmousi-otmmd-iter100/x_100.mat')
    #initial_value = solver.ModelParameters(m, {'C': C0})
    #initial_value.data = initial_value.with_padding(padding_mode='edge').data
    #initial_value.padded = True

    #initial_value.data = C_ot_load['data']
    #C_ot = initial_value.without_padding().data

    #initial_value = solver.ModelParameters(m, {'C': C_ot})
    #########################################################################
    initial_value = solver.ModelParameters(m, {'C': C0})
    initial_value.data = initial_value.with_padding(padding_mode='edge').data
    initial_value.padded = True

    # Execute inversion algorithm
    tt = time.time()

    nsteps = 100
    result = invalg(shots, initial_value, nsteps,
                        line_search=line_search,
                        status_configuration=status_configuration, verbose=True, write=True)

    initial_value.data = result.C
    C_cut = initial_value.without_padding().data

####################################################################################################
    # Save wavefield
    inverted_model = solver.ModelParameters(m,{'C': C_cut})
    generate_seismic_data(shots, solver, inverted_model)
    wavefield_inverted = comm.gather(shots[0].receivers.data, root=0)

    # SaveData
    if rank == 0:
        model = C_cut.reshape(m.shape(as_grid=True)).transpose()

        vals = list()
        for k,v in list(invalg.objective_history.items()):
            vals.append(v)
        obj_vals = np.array(vals)

        initial_value.data = invalg.gradient_history[0].data
        gradient_iter1 = initial_value.without_padding().data.reshape(m.shape(as_grid=True)).transpose()
        initial_value.data = invalg.gradient_history[int(nsteps/2)].data
        gradient_iter2 = initial_value.without_padding().data.reshape(m.shape(as_grid=True)).transpose()
        initial_value.data = invalg.gradient_history[int(nsteps-1)].data
        gradient_iter3 = initial_value.without_padding().data.reshape(m.shape(as_grid=True)).transpose()

        output = {'inverted': model,
                  'true': C.reshape(m.shape(as_grid=True)).transpose(),
                  'initial': C0.reshape(m.shape(as_grid=True)).transpose(),
                  'obj': obj_vals,
                  'dt': solver.dt,
                  'wavefield_true': wavefield_true,
                  'wavefield_initial': wavefield_initial,
                  'wavefield_inverted': wavefield_inverted,
                  'gradient_0': gradient_iter1,
                  'gradient_middle': gradient_iter2,
                  'gradient_end': gradient_iter3,
                 }

        sio.savemat(ExpDir + '/output.mat', output)
