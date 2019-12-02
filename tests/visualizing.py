"""
Python script to plot results of pySIT run.

Use
---

        python visualize.py <directory>

where directory contains the results of a run.

"""
import os
import sys
from datetime import datetime
import argparse
import re

import numpy as np
import pickle
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal

from pysit import *

plt.rcParams["font.size"] = '22'
plt.rcParams['figure.figsize'] = 20,10
plt.rcParams['lines.linewidth'] = 4


def save_convergence_plot(conv, output_dir: str):
    """
    Plot FWI convergence curve
    """
    fig = plt.figure()
    plt.semilogy(conv[0]/np.max(conv[0])) 
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    plt.title('FWI convergence curve')                                    
    plt.grid(True)
    fig.savefig(os.path.join(output_dir, 'conv.png')) 


def save_velocity_profile_plot(model, kind: str, mesh, clim: list, output_file: str):
    """
    Save model into file.

    output_file: str
        Absolute path/name of the file
    """
    fig, ax = plt.subplots(figsize=(32,12))
    aa=vis.plot(model.T, mesh, clim=clim)
    title = kind + ' model'
    ax.set(xlabel='Offset [km]', ylabel='Depth [km]', title=title)
    ax.set_aspect('auto')
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(aa)
    cbar.ax.set_ylabel('Velocity [km/s]')
    fig.savefig(output_file)


def save_gradient_plot(gradient, mesh, output_dir: str):
    """
    Save gradient.png into output_dir.
    """
    clim = gradient.min(),gradient.max()
    fig, ax = plt.subplots(figsize=(32,12))
    aa=vis.plot(gradient, mesh, clim=clim)
    ax.set(xlabel='Offset [km]', ylabel='Depth [km]', title='Gradient - iter#1')
    ax.set_aspect('auto')
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(aa)
    cbar.ax.set_ylabel('Velocity [km/s]')
    fig.savefig(os.path.join(output_dir, 'gradient.png'))


def save_wavefield_plot(wavefield, kind: str, ext: list, clim: list, output_dir: str):
    """
    Save wavefield into file.
    """
    fig, ax = plt.subplots(figsize=(32, 12))
    aa = plt.imshow(wavefield, interpolation='nearest', aspect='auto', cmap='seismic', clim=clim,
                    extent=ext)

    title = kind + ' wavefield'
    ax.set(xlabel='Offset [km]', ylabel='Time [s]', title=title)
    ax.set_aspect('auto')
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(aa)
    cbar.ax.set_ylabel('Amplitude')
    filepath = os.path.join(output_dir,'wavefield-' + kind + '.png')
    fig.savefig(filepath)



def save_wavefields_receiver_plot(wfrs, tt, output_dir: str):
    """
    Save wavefields receiver into output_dir.
    """
    fig = plt.figure()
    ntr = int(np.shape(wfrs['True'])[1]/3 + 1)
    #tt = (np.arange(np.shape(wfrs['True'])[0]) * dt).transpose()
    plt.plot(tt, wfrs['True'][:, ntr], 'b', label='data-observed')
    plt.plot(tt, wfrs['Initial'][:, ntr], 'g', label='data-initial')
    plt.plot(tt, wfrs['Inverted'][:, ntr], 'r', label='data-inverted')
    plt.xlabel('Time [s]')
    plt.title('Wavefield-receiver-'+str(ntr))
    plt.grid(True)
    plt.legend()
    filepath = os.path.join(output_dir, 'wavefield_receiver_'+str(ntr)+'.png')
    fig.savefig(filepath)


def save_wavefields_time_plot(wfrs, t_range, output_dir:str):
    """
    Save wavefields time into output_dir x, t
    """
    fig = plt.figure()
    ntr = int(np.shape(wfrs['True'])[0]/t_range[1] + 1)
    aaaa = np.round(ntr * dt, 1)
    xx = np.arange(np.shape(wfrs['True'])[1]) * m.x.delta
    plt.plot(xx, wfrs['True'][ntr, :], 'b', label='data-observed')
    plt.plot(xx, wfrs['Initial'][ntr, :], 'g', label='data-initial')
    plt.plot(xx, wfrs['Inverted'][ntr, :], 'r', label='data-inverted')
    plt.xlabel('Receivers [km]')
    plt.title('Wavefield-time-'+str(aaaa)+'s')
    plt.grid(True)
    plt.legend()
    filepath = os.path.join(output_dir, 'wavefield_time_'+str(aaaa)+'s.png')
    fig.savefig(filepath)


def save_adjoint_source_plot(adj, ext: list, clim: list, output_dir: str):
    """
    Save adjoint source plot in output_dir
    """
    fig, ax = plt.subplots(figsize=(32,12))
    aa=plt.imshow(adj, interpolation='nearest', aspect='auto', cmap='gray', clim=clim,
            extent=ext)
    ax.set(xlabel='Offset [km]', ylabel='Time [s]', title='Adjoint sources of the first iteration -- L2')
    ax.set_aspect('auto')
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(aa)
    cbar.ax.set_ylabel('Amplitude')
    filepath = os.path.join(output_dir, 'adjoint_source_l2.png')
    fig.savefig(filepath)


def parse_cmd_line() -> str:
    """
    Returns the absolute path of the directory
    passed in command line.
    """

    cmd_line = argparse.ArgumentParser(
        description='Python script to plot results of pySIT run '+
                    'stored in a given directory.',
        usage='''python visualize.py <directory>''')
    cmd_line.add_argument('directory', help='Results directory')

    args = cmd_line.parse_args()
    
    if not os.path.exists(args.directory):
        raise IOError("Results directory doesn't exist.")

    return os.path.abspath(args.directory)


def save_velocity_traces(V: dict, X, Z, n_traces: int, file_base_name: str, output_dir: str):
        """
        Compute equidistant traces along x-axis of velocity profiles
        and save them on the same plot in a file located in output_dir.

        Parameters
        ----------
        V : dict
            Velocity profiles
        X : 1D numpy array
            x coordinates samples
        Z : 1D numpy array
            z coordinates samples
        n_traces : int
            number of equidistant traces along x-axis to save
        file_base_name : str
            Prefix of the file name, x coordinate of the trace is
            appended.
        output_dir : str
            Directory where figures are saved
        
        """
        nx = len(X)

        # Compute traces indices along x
        idx_traces = np.linspace(0, nx-1, n_traces, dtype=int)

        for idx in idx_traces:
            fig = plt.figure(figsize=(16,20))

            # x value corresponding to idx
            x = np.round(X[idx],2)

        #     for vm in velocity_models:
        #         plt.plot(velocity_models[vm][:,idx],Z, label='V-trace-'+vm)
            if np.shape(list(V.keys())[0].split('/x_'))[0]==2:
                plt.plot(velocity_models['True'][:,idx],Z, label='V-trace-true')
                for vm in V:
                    im = vm.split('/x_')[1]
                    ni = int(len(V)/9)
                    if int(im)%ni == 1:
                        plt.plot(V[vm][:,idx],Z, label='V-trace-iter#'+im)
            else:
                for vm in V:
                    plt.plot(V[vm][:,idx],Z, label='V-trace-'+vm.split('/x_')[0])

            plt.xlabel('velocity [km/s]')
            plt.ylabel('Depth [km]')
            plt.ylim(max(Z),min(Z))
            plt.grid(True)
            plt.legend()
            filepath = os.path.join(output_dir, file_base_name+str(x)+'.png')
            fig.savefig(filepath)


def create_plot_dir(base_dir: str) -> str:
    """
    Create directory to store plots in the base_dir
    and returns it.

    Parameters
    ----------
    base_dir: str
        Directory where results are
    """
    time_str = datetime.now().strftime('%Y%b%d-%H%M%S') 
    plot_dir = os.path.join(res_dir, 'fig_'+time_str)
#     plot_dir = os.path.join(res_dir, 'plot')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    return plot_dir


def get_velocity_profiles(base_dir: str, pattern: str, mesh):
    """
    Parse base_dir to get the velocity profiles.
    Store them into a dict which keys are the
    iterations numbers.

    Parameters
    ----------
    base_dir: str
        Directory where results are
    pattern: str
        Regular expression that matches velocity profiles file
        names from different iterations
    """
    compiled_pattern = re.compile(pattern)
    # List of results files mathching the reg exp.
    res_file_list = [os.path.join(base_dir,x) for x in os.listdir(base_dir) if compiled_pattern.match(x)]
    res_file_list.sort()

    # Iteration number list
    it_nbrs = ['iter_'+x.split('.')[0][2:] for x in res_file_list]
    it_nbrs.sort()

    X = [loadmat(x) for x in res_file_list]
    V = [np.reshape(x['data'],mesh.shape(as_grid=True)).transpose() for x in X]
    V_dict = dict(zip(it_nbrs, V))

    return V_dict


if __name__ == "__main__":

    res_dir = parse_cmd_line()
    print("Parsing results directory:\n", res_dir)
    
    p_dir = create_plot_dir(res_dir)

    #### Load data
    with open(os.path.join(res_dir,'mesh.p'),'rb') as f:
        m = pickle.load(f)

    output = loadmat(os.path.join(res_dir, 'output.mat'))

    # Generate domain coordinate samples
    x_range = output['x_range'][0]
    z_range = output['z_range'][0]
    nz, nx = np.shape(output['true'])
    X_coord = np.linspace(x_range[0], x_range[1], nx)
    Z_coord = np.linspace(z_range[0], z_range[1], nz)

    #### Velocity models c(x, z)
    # True = direct problem results
    # Initial = Initialization 
    # Inverted = Last iteration result
    velocity_models = {
        'True': output['true'],
        'Initial': output['initial'],
        'Inverted': output['inverted']
    }
    clim = velocity_models['True'].min(),velocity_models['True'].max()
    for vm in velocity_models:
        abs_path_name = os.path.join(p_dir,'model-'+vm+'.png')
        save_velocity_profile_plot(velocity_models[vm], vm, m, clim, abs_path_name)

    #### Velocity traces
    n_tr = 5
    
    # Velocity traces from normal models
    save_velocity_traces(velocity_models, X_coord, Z_coord, n_tr, 'velocity_trace_', p_dir)

    # Velocity traces at different iterations
    res_name = '^x_[0-9]+.mat'
    V = get_velocity_profiles(res_dir, res_name, m)
    save_velocity_traces(V, X_coord, Z_coord, n_tr, 'iter_velocity_trace_', p_dir)

    #### Convergence
    save_convergence_plot(output['conv'], p_dir)

    #### Wavefields w(t, x)
    # True = direct problem results
    # Initial = Initialization 
    # Inverted = Last iteration result
    wavefields = {
        'True': output['wavefield_true'],
        'Initial': output['wavefield_initial'],
        'Inverted': output['wavefield_inverted']
    }
    # Re-sampling data on time
    nt = np.shape(wavefields['True'])[0]
    wavefields['Inverted'] = signal.resample(wavefields['Inverted'], nt)
    wavefields['Initial'] = signal.resample(wavefields['Initial'], nt)

    t_range = output['t_range'][0]
    T_coord = np.linspace(t_range[0], t_range[1], nt)
    extent = [min(X_coord), max(X_coord), max(T_coord), min(T_coord)]
    clim=[-.05, .05]
    for wf in wavefields:
        save_wavefield_plot(wavefields[wf], wf, extent, clim, p_dir)

    save_wavefield_plot(wavefields['True']-wavefields['Inverted'], 'diff', extent, clim, p_dir)

    #### Gradient
    save_gradient_plot(output['gradient'].T, m, p_dir)

    #### Wavefields - receiver
    save_wavefields_receiver_plot(wavefields, T_coord, p_dir)

    #### Wavefields -time
    #save_wavefields_time_plot(wavefields, trange, p_dir)

    ### Adjoint source
    adj_l2 = wavefields['True'] - wavefields['Initial']
    clim = np.min(adj_l2),np.max(adj_l2)
    save_adjoint_source_plot(adj_l2, extent, clim, p_dir)
