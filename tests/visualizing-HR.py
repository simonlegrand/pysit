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
mpl.use('Agg')
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


def save_model_plot(model, kind: str, mesh, clim: list, output_dir: str):
    """
    Save model into file.
    """
    fig, ax = plt.subplots(figsize=(32,12))
    aa=vis.plot(model.T, mesh, clim=clim)
    ax.set(xlabel='Offset [km]', ylabel='Depth [km]', title='Initial Model')
    ax.set_aspect('auto')
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(aa)
    cbar.ax.set_ylabel('Velocity [km/s]')
    filepath = os.path.join(output_dir,'model-' + kind + '.png')
    fig.savefig(filepath)


def save_gradient_plot(gradient, output_dir: str):
    """
    Save gradient.png into output_dir.
    """
    clim = gradient.min(),gradient.max()
    fig, ax = plt.subplots(figsize=(32,12))
    aa=vis.plot(gradient.T, m, clim=clim)
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


def save_wavefields_receiver_plot(wfrs, output_dir: str):
    """
    Save wavefields receiver into output_dir.
    """
    fig = plt.figure()
    ntr = int(np.shape(wavefields['true'])[1]/3 + 1)
    tt = (np.arange(np.shape(wavefields['true'])[0]) * dt).transpose()
    plt.plot(tt, wavefields['true'][:, ntr], 'b', label='data-observed')
    plt.plot(tt, wavefields['initial'][:, ntr], 'g', label='data-initial')
    plt.plot(tt, wavefields['inverted'][:, ntr], 'r', label='data-inverted')
    plt.xlabel('Time [s]')
    plt.title('Wavefield-receiver-'+str(ntr))
    plt.grid(True)
    plt.legend()
    filepath = os.path.join(output_dir, 'wavefield_receiver_'+str(ntr)+'.png')
    fig.savefig(filepath)


def save_wavefields_time_plot(wfrs, t_range, output_dir:str):
    """
    Save wavefields time into output_dir.
    """
    fig = plt.figure()
    ntr = int(np.shape(wavefields['true'])[0]/t_range[1] + 1)
    aaaa = np.round(ntr * dt, 1)
    xx = np.arange(np.shape(wavefields['true'])[1]) * m.x.delta
    plt.plot(xx, wavefields['true'][ntr, :], 'b', label='data-observed')
    plt.plot(xx, wavefields['initial'][ntr, :], 'g', label='data-initial')
    plt.plot(xx, wavefields['inverted'][ntr, :], 'r', label='data-inverted')
    plt.xlabel('Receivers [km]')
    plt.title('Wavefield-time-'+str(aaaa)+'s')
    plt.grid(True)
    plt.legend()
    filepath = os.path.join(output_dir, 'wavefield_time_'+str(aaaa)+'s.png')
    fig.savefig(filepath)


def save_velocity_profiles_plot(vms, mesh, output_dir: str):
    """
    Save velocity profiles
    """
    trc = int(5)
    delta_x = (mesh.x.n-1)/(2*trc)

    for i in range(trc):
        fig = plt.figure(figsize=(16,20))
        ntrc = (int((mesh.x.n-1)/2 + i*delta_x)) + 1

        zz = np.arange(np.shape(vms['initial'])[0]) * mesh.z.delta
        plt.plot(vms['initial'][:, ntrc], zz, 'g', label='Vp-initial')
        plt.plot(vms['true'][:, ntrc], zz, 'b', label='Vp-true')
        plt.plot(vms['inverted'][:, ntrc], zz, 'r', label='Vp-inverted')
        plt.xlabel('velocity [km/s]')
        plt.ylabel('Depth [km]')
        plt.title('Traces-'+str(ntrc)+' comparison along z-axis')
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.grid(True)
        plt.legend()
        filepath = os.path.join(output_dir,'velocity_file_'+str(ntrc)+'.png')
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


def create_plot_dir(base_dir: str) -> str:
    """
    Create directory to store plots in the base_dir
    and returns it.

    base_dir: str
        Directory where results are
    """
    time_str = datetime.now().strftime('%Y%b%d-%H%M%S') 
    plot_dir = os.path.join(res_dir, 'fig_'+time_str)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    return plot_dir


if __name__ == "__main__":

    res_dir = parse_cmd_line()
    print("Parsing results directory:\n", res_dir)
    
    p_dir = create_plot_dir(res_dir)

    #### Load data
    with open(os.path.join(res_dir,'mesh.p'),'rb') as f:
        m = pickle.load(f)

    output = loadmat(os.path.join(res_dir, 'output.mat'))

    #### Convergence
    save_convergence_plot(output['conv'], p_dir)

    #### Velocity models
    velocity_models = {
        'true': output['true'],
        'initial': output['initial'],
        'inverted': output['inverted']
    }
    clim = velocity_models['true'].min(),velocity_models['true'].max()
    for vm in velocity_models:
        save_model_plot(velocity_models[vm], vm, m, clim, p_dir)

    #### Wavefields
    wavefields = {
        'true': output['wavefield_true'],
        'initial': output['wavefield_initial'],
        'inverted': output['wavefield_inverted']
    }
    # Re-sampling data on time
    shape_dobs = np.shape(wavefields['true'])
    wavefields['inverted'] = signal.resample(wavefields['inverted'], shape_dobs[0])
    wavefields['initial'] = signal.resample(wavefields['initial'], shape_dobs[0])

    x_length = output['x_length']
    trange = output['t_record'][0]
    dt = trange[1]/shape_dobs[0]
    t_smp = np.linspace(trange[0], trange[1], shape_dobs[0])
    extent = [0.0, x_length, t_smp[-1], 0.0]
    clim=[-.05, .05]
    for wf in wavefields:
        save_wavefield_plot(wavefields[wf], wf, extent, clim, p_dir)

    clim2=[-4e-2, 4e-4]
    save_wavefield_plot(wavefields['true']-wavefields['inverted'], 'diff', extent, clim2, p_dir)

    #### Gradient
    save_gradient_plot(output['gradient'], p_dir)

    #### Wavefields - receiver
    save_wavefields_receiver_plot(wavefields, p_dir)

    #### Wavefields -time
    save_wavefields_time_plot(wavefields, trange, p_dir)

    #### Velocity profiles
    save_velocity_profiles_plot(velocity_models, m, p_dir)

    #### Adjoint source
    adj_l2 = wavefields['true'] - wavefields['initial']
    clim = np.min(adj_l2),np.max(adj_l2)
    save_adjoint_source_plot(adj_l2, extent, clim, p_dir)


    res_name_re = re.compile('^x_[0-9]+.mat')

    # List of results files
    res_file_list = [x for x in os.listdir(res_dir) if res_name_re.match(x)]

    # Number of iterations
    N = len(res_file_list)

    # Iteration number list
    it_nbrs = [x.split('.')[0][2:] for x in res_file_list]
