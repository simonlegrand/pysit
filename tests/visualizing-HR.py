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



# Build path for saving figures
RootDir = ''
SubDir = str(sys.argv[1])
#######################################################################################################################

ExpDir = RootDir + SubDir

time_str = datetime.now().strftime('%Y%b%d-%H%M%S') 
path_fig = (ExpDir + '/fig_' + time_str)
if not os.path.exists(path_fig):
    os.makedirs(path_fig)

###################################### Loading data ############################
with open('mesh.p','rb') as f:
    m = pickle.load(f)

#####################
output = loadmat(ExpDir + '/output.mat')
""" 
    output = {'conv': conv_vals,
              'inverted': C_inverted,
              'true': C.reshape(m.shape(as_grid=True)).transpose(),
              'initial': C0.reshape(m.shape(as_grid=True)).transpose(),
              'wavefield_true': wavefield_true,
              'wavefield_initial': wavefield_initial,
              'wavefield_inverted': wavefield_inverted,
              'gradient': gradient,
              'x_length': d.x.rbound,
              't_record': trange,
              }
"""
conv = output['conv']
velocity_model_true = output['true']
velocity_model_initial = output['initial']
velocity_model_inverted = output['inverted']

velocity_models = {
    'true': output['true'],
    'initial': output['initial'],
    'inverted': output['inverted']
}

wavefields = {
    'true': output['wavefield_true'],
    'initial': output['wavefield_initial'],
    'inverted': output['wavefield_inverted']
}

gradient = output['gradient']
x_length = output['x_length']
trange = output['t_record'][0]

# Re-sampling data on time
shape_dobs = np.shape(wavefields['true'])
wavefields['inverted'] = signal.resample(wavefields['inverted'], shape_dobs[0])
wavefields['initial'] = signal.resample(wavefields['initial'], shape_dobs[0])

dt = trange[1]/shape_dobs[0]
t_smp = np.linspace(trange[0], trange[1], shape_dobs[0])

##################### Convergence ########################
fig1 = plt.figure()
plt.semilogy(conv[0]/np.max(conv[0])) 
plt.xlabel('Iteration')
plt.ylabel('Objective value')
plt.title('FWI convergence curve')                                    
plt.grid(True)
fig1.savefig(path_fig+'/conv.png')  

#################### Model - initial, true, inverted ############

def save_model_plot(model, kind: str, mesh, clim, path_fig: str):
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
    filepath = os.path.join(path_fig,'model-' + kind + '.png')

    fig.savefig(filepath)

clim = velocity_models['true'].min(),velocity_models['true'].max()

for vm in velocity_models:
    save_model_plot(velocity_models[vm], vm, m, clim, path_fig)


#################### Gradient ####################################
clim = gradient.min(),gradient.max()
fig3, ax2 = plt.subplots(figsize=(32,12))
aa=vis.plot(gradient.T, m, clim=clim)
ax2.set(xlabel='Offset [km]', ylabel='Depth [km]', title='Gradient - iter#1')
ax2.set_aspect('auto')
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig3.colorbar(aa)
cbar.ax.set_ylabel('Velocity [km/s]')
fig3.savefig(path_fig+'/gradient.png')

#################### Wavefields - initial, true, inverted, diff ##########################

def save_wavefield_plot(wavefield, kind: str, clim, path_fig: str):
    """
    Save wavefield into file.
    """
    fig, ax = plt.subplots(figsize=(32, 12))
    aa = plt.imshow(wavefield, interpolation='nearest', aspect='auto', cmap='seismic', clim=clim,
                    extent=[0.0, x_length, t_smp[-1], 0.0])

    title = kind + ' wavefield'
    ax.set(xlabel='Offset [km]', ylabel='Time [s]', title=title)
    ax.set_aspect('auto')
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = fig.colorbar(aa)
    cbar.ax.set_ylabel('Amplitude')
    filepath = os.path.join(path_fig,'wavefield-' + kind + '.png')

    fig.savefig(filepath)

clim=[-.05, .05]
clim2=[-4e-2, 4e-4]

for wf in wavefields:
    save_wavefield_plot(wavefields[wf], wf, clim, path_fig)

diff_wavefield = wavefields['true']-wavefields['inverted']
save_wavefield_plot(diff_wavefield, 'diff', clim2, path_fig)

############################## Wavefields - receiver #################################
fig5 = plt.figure()
ntr = int(np.shape(wavefields['true'])[1]/3 + 1)
tt = (np.arange(np.shape(wavefields['true'])[0]) * dt).transpose()
plt.plot(tt, wavefields['true'][:, ntr], 'b', label='data-observed')
plt.plot(tt, wavefields['initial'][:, ntr], 'g', label='data-initial')
plt.plot(tt, wavefields['inverted'][:, ntr], 'r', label='data-inverted')
plt.xlabel('Time [s]')
plt.title('Wavefield-receiver-'+str(ntr))
plt.grid(True)
plt.legend()
fig5.savefig(path_fig+'/wavefield_receiver_'+str(ntr)+'.png')

############################## Wavefields - time #################################
fig6 = plt.figure()
ntr = int(np.shape(wavefields['true'])[0]/trange[1] + 1)
aaaa = np.round(ntr * dt, 1)
xx = np.arange(np.shape(wavefields['true'])[1]) * m.x.delta
plt.plot(xx, wavefields['true'][ntr, :], 'b', label='data-observed')
plt.plot(xx, wavefields['initial'][ntr, :], 'g', label='data-initial')
plt.plot(xx, wavefields['inverted'][ntr, :], 'r', label='data-inverted')
plt.xlabel('Receivers [km]')
plt.title('Wavefield-time-'+str(aaaa)+'s')
plt.grid(True)
plt.legend()
fig6.savefig(path_fig+'/wavefield_time_'+str(aaaa)+'s.png')

############################## verlocity profiles - initial, true, inverted #################################
trc = int(5)
delta_x = (m.x.n-1)/(2*trc)

for i in range(trc):
        fig = plt.figure(figsize=(16,20))
        ntrc = (int((m.x.n-1)/2 + i*delta_x)) + 1

        zz = np.arange(np.shape(velocity_model_initial)[0]) * m.z.delta
        plt.plot(velocity_model_initial[:, ntrc], zz, 'g', label='Vp-initial')
        plt.plot(velocity_model_true[:, ntrc], zz, 'b', label='Vp-true')
        plt.plot(velocity_model_inverted[:, ntrc], zz, 'r', label='Vp-inverted')
        plt.xlabel('velocity [km/s]')
        plt.ylabel('Depth [km]')
        plt.title('Traces-'+str(ntrc)+' comparison along z-axis')
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.grid(True)
        plt.legend()
        fig.savefig(path_fig+'/velocity_file_'+str(ntrc)+'.png')

#################### Adjoint source -- L2 ####################################
adj_l2 = wavefields['true'] - wavefields['initial']
clim = np.min(adj_l2),np.max(adj_l2)

fig, ax2 = plt.subplots(figsize=(32,12))
aa=plt.imshow(adj_l2, interpolation='nearest', aspect='auto', cmap='gray', clim=clim,
        extent=[0.0, x_length, t_smp[-1], 0.0])
ax2.set(xlabel='Offset [km]', ylabel='Time [s]', title='Adjoint sources of the first iteration -- L2')
ax2.set_aspect('auto')
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig.colorbar(aa)
cbar.ax.set_ylabel('Amplitude')
fig.savefig(path_fig+'/adjoint_source_l2.png')
