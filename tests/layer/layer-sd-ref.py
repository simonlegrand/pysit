# Std import block
import time
import pickle
import numpy as np

import copy
import sys
import scipy.io as sio

from pysit import *
from pysit.gallery.layered_medium import four_layered_medium
from pysit.util.io import *
# from pysit.util.compute_tools import *
from pysit.util.parallel import *

from mpi4py import MPI

if __name__ == '__main__':
    # Setup
    RootDir = '/scratch/miyu/results-pysit/'
    SubDir = ''
    # ExpDir = RootDir + SubDir

    ExpDir = '.'

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    pwrap = ParallelWrapShot()

    if rank == 0:
        ttt = time.time()
        sys.stdout.write('Four-layered medium \n')

    # Set up domain, mesh and velocity model
    model_param = { 'x_length'          : 15.0,
                    'z_depth'           : 6.0,
                    'velocity'          : (1.7, 3.0, 3.5, 4.0),
                    'layer_thickness'   : (1.0, 1.0, 1.5, 2.5),
                  }

    C, C0, m, d = four_layered_medium(model_param=model_param,
                                      dx = 0.05, dz = 0.05,
                                      # initial_model_style='gradient', 
                                      # initial_config={'gradient_slope':0.85},
                                      initial_model_style='layer',
                                      initial_config={'initial_velocity' : (1.7, 3.0),
                                                      'initial_thickness': (1.0, 5.0)},
                                     )

    # Set up shots
    zmin = d.z.lbound
    zmax = d.z.rbound
    zpos = 0.05 * 1.0

    Nshots = size
    Nreceivers = 301
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
    t_range = (0.0,4.0)

    solver = ConstantDensityAcousticWave(m,
                                         spatial_accuracy_order=6,
                                         trange=t_range,
                                         kernel_implementation='cpp',
                                         max_C=None) # The dt is automatically fixed for given max_C (velocity)

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

    base_model = solver.ModelParameters(m,{'C': C})
    tt = time.time()
    generate_seismic_data(shots, solver, base_model)
    wavefield_true = comm.gather(shots[0].receivers.data, root=0)

    if rank==0:
        # ## Save initial and true wavefield
        # tmp_save_wavefield = {'wavefield_true': wavefield_true,
        #                       'wavefield_initial': wavefield_initial,
        #                      }
        # sio.savemat(ExpDir + '/wavefields.mat', tmp_save_wavefield)

        # ## Save initial and true model
        # nt = m.shape(as_grid=True)
        # nt = (nt[1], nt[0])
        # dt = (m.parameters[1].delta, m.parameters[0].delta)
        # ot = (d.z.lbound, d.x.lbound)

        # write_data(ExpDir + '/model-true.mat', C.reshape(m.shape(as_grid=True)).transpose(), ot, dt, nt)
        # write_data(ExpDir + '/model-initial.mat', C0.reshape(m.shape(as_grid=True)).transpose(), ot, dt, nt)
        # ###################################
        sys.stdout.write('{1}:Data generation: {0}s\n'.format(time.time()-tt,rank))
    sys.stdout.flush()

    comm.Barrier()

    if rank == 0:
        tttt = time.time()-ttt
        sys.stdout.write('Total wall time: {0}\n'.format(tttt))
        sys.stdout.write('Total wall time/shot: {0}\n'.format(tttt/Nshots))

    ############# Set up objective function ##############
    ot_param = { 'n_receivers'                  : Nreceivers,
                 'sinkhorn_iterations'          : 10000,
                 'sinkhorn_tolerance'           : 1.0e-9,
                 'successive_over_relaxation'   : 1.,
                 'sinkhorn_initialization'      : True,  # True or False
                 'epsilon_kl'                   : 1e-2,
                 'lamb_kl'                      : 0.1,
                 't_scale'                      : 10.0,
                 'x_scale'                      : 10.0,
                 'data_process_type'            : 'IDsT',   # 'IDsT'; 'DsT'; 'IDcT'; 'TDs'; 'ITDc'; 'TI'
                 'trans_func_type'              : 'smooth_max',  # smooth_max//exp_max//exp_sum//exp_orig//exp_jd//square//id;id2// 
                 'trans_func_factor'            : 1.0e-5,    #  epsilon_sm for the smooth_max // k for exp(k*signal)
                 'nt_NormSpace'                 : 128,
                 'dc_factor'                    : None,
                 'muteInit'                     : None, #wavefield_initial, # None
                 'padding'                      : None,  # [nl, nr]
                 'velocity_bound'               : None,  # [1.5, 4.0], # None
                 'filter_op'                    : False,  # True or False
                 'freq_band'                    : [5.0, 30.0],
                 'noise_factor'                 : None, # None or a scalar 0.01
               }

    # Sinkhorn Divergence objective function
    if rank == 0:
        print('Sinkhorn Divergence...')
        print('Sinkhorn Divergence parameters setting:')
        print('sinkhorn-initialization = %s' %ot_param['sinkhorn_initialization'])
        print('sinkhorn-epsilon-kl = %.2f' %ot_param['epsilon_kl'])
        print('sinkhorn-lamb-kl = %.1f' %ot_param['lamb_kl'])
        print('sinkhorn-tscale = %.1f' %ot_param['t_scale'])
        print('sinkhorn-xscale = %.1f' %ot_param['x_scale'])
        print('data-process-type = %s' %ot_param['data_process_type'])
        print('trans-func-type = %s' %ot_param['trans_func_type'])
        print('trans-func-factor = %.1e' %ot_param['trans_func_factor'])
        print('nt-NormSpace = %d' %ot_param['nt_NormSpace'])
        if ot_param['dc_factor'] is not None:
            print('dc_factor = %d' %ot_param['dc_factor'])
        if ot_param['muteInit'] is not None:
            print('muteInit = True')
        if ot_param['padding'] is not None:
            print('padding = %s' %ot_param['padding'])
        print('velocity-bound = %s' %ot_param['velocity_bound'])
        print('noise-factor = %s' %ot_param['noise_factor'])
        print('filter-operation = %s' %ot_param['filter_op'])
        if ot_param['filter_op'] is True:
            print('filter-frequency-band = %s' %ot_param['freq_band'])

    objective = SinkhornDivergence(solver, ot_param=ot_param, parallel_wrap_shot=pwrap)   

    # # Least-squares objective function
    # if rank == 0:
    #     print('Least-squares...')
    #     print('Parameters setting:')
    #     print('data-process-type = %s' %ot_param['data_process_type'])
    #     print('trans-func-type = %s' %ot_param['trans_func_type'])
    #     print('trans-func-factor = %.1e' %ot_param['trans_func_factor'])
    #     print('nt-NormSpace = %d' %ot_param['nt_NormSpace'])
    #     print('dc_factor = %d' %ot_param['dc_factor'])
    #     if ot_param['muteInit'] is not None:
    #         print('muteInit = True')
    #     if ot_param['padding'] is not None:
    #         print('padding = %s' %ot_param['padding'])
    #     print('velocity-bound = %s' %ot_param['velocity_bound'])
    #     print('noise-factor = %s' %ot_param['noise_factor'])
    #     print('filter-operation = %s' %ot_param['filter_op'])
    #     if ot_param['filter_op'] is True:
    #         print('filter-frequency-band = %s' %ot_param['freq_band'])
            
    # objective = TemporalLeastSquares(solver, ot_param=ot_param, parallel_wrap_shot=pwrap)

    # # Envelope objective function
    # print('Envelope...')
    # objective = TemporalEnvelope(solver, envelope_power=2.0, filter_op=filter_op1, imaging_period=1, parallel_wrap_shot=pwrap)

    # # Cross-correlation objective function
    # print('Cross-correlation...')
    # objective = TemporalCorrelate(solver, filter_op=filter_op1, imaging_period=1, parallel_wrap_shot=pwrap)   

    # # Optimal transportation objective function with linear transformation
    # print('Ot with linear transformation...')
    # objective = TemporalOptimalTransport(solver, filter_op=filter_op1, imaging_period=1, transform_mode='linear', c_ratio=2.0, parallel_wrap_shot=pwrap)

    # # Optimal transportation objective function with quadratic transformation
    # print('Ot with quadratic transformation...')
    # objective = TemporalOptimalTransport(solver, filter_op=filter_op1, imaging_period=1, transform_mode='quadratic', parallel_wrap_shot=pwrap)

    # # Optimal transportation objective function with absolute transformation
    # print('Ot with absolute transformation...')
    # objective = TemporalOptimalTransport(solver, filter_op=filter_op1, imaging_period=1, transform_mode='absolute', parallel_wrap_shot=pwrap)

    # # Optimal transportation objective function with exponential transformation
    # print('Ot with exponential transformation...')
    # objective = TemporalOptimalTransport(solver, filter_op=filter_op1, imaging_period=1, transform_mode='exponential', exp_a=1.0, parallel_wrap_shot=pwrap)  


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

    if rank==0:
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
    #initial_value.data = initial_value.with_padding(padding_mode='edge').data
    #initial_value.padded = True

    # Execute inversion algorithm
    tt = time.time()

    nsteps = 300
    result = invalg(shots, initial_value, nsteps,
                        line_search=line_search,
                        status_configuration=status_configuration, verbose=True, write=True)

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
        with open('mesh.p', 'wb') as f:
            pickle.dump(m, f)

        conv_vals = np.array([v for k,v in list(invalg.objective_history.items())])

        initial_value.data = invalg.gradient_history[0].data
        gradient = initial_value.without_padding().data.reshape(m.shape(as_grid=True)).transpose()

        ns = int(np.shape(wavefield_true)[0]/2)

        if ot_param['noise_factor'] is None:
            noise = 0
        else:
            noise = ot_param['noise_factor']

        output = {'conv':                 conv_vals,
                  'inverted':             C_inverted,
                  'true':                 C.reshape(m.shape(as_grid=True)).transpose(),
                  'initial':              C0.reshape(m.shape(as_grid=True)).transpose(),
                  'wavefield_true':       wavefield_true[ns],
                  'wavefield_initial':    wavefield_initial[ns],
                  'wavefield_inverted':   wavefield_inverted[ns],
                  'gradient':             gradient,
                  'x_range':              [d.x.lbound, d.x.rbound],
                  'z_range':              [d.z.lbound, d.z.rbound],
                  't_range':              t_range,
                  'obj_name':             objective.name(),
                  'noise_factor':         noise,
                  }

        sio.savemat(ExpDir + '/output.mat', output)
