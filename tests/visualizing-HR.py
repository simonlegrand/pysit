# Std import block
import os
import sys
from datetime import datetime

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
wavefield_true = output['wavefield_true']
wavefield_initial = output['wavefield_initial']
wavefield_inverted = output['wavefield_inverted']
gradient = output['gradient']
x_length = output['x_length']
trange = output['t_record'][0]

# Re-sampling data on time
shape_dobs = np.shape(wavefield_true)
wavefield_inverted = signal.resample(wavefield_inverted, shape_dobs[0])
wavefield_initial = signal.resample(wavefield_initial, shape_dobs[0])

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
clim = velocity_model_true.min(),velocity_model_true.max()
fig2, ax2 = plt.subplots(figsize=(32,12))
aa=vis.plot(velocity_model_initial.T, m, clim=clim)
ax2.set(xlabel='Offset [km]', ylabel='Depth [km]', title='Initial Model')
ax2.set_aspect('auto')
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig2.colorbar(aa)
cbar.ax.set_ylabel('Velocity [km/s]')
fig2.savefig(path_fig+'/model-initial.png')

fig22, ax2 = plt.subplots(figsize=(32,12))
aa=vis.plot(velocity_model_true.T, m, clim=clim)
ax2.set(xlabel='Offset [km]', ylabel='Depth [km]', title='True Model')
ax2.set_aspect('auto')
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig22.colorbar(aa)
cbar.ax.set_ylabel('Velocity [km/s]')
fig22.savefig(path_fig+'/model-true.png')

fig23, ax2 = plt.subplots(figsize=(32,12))
aa=vis.plot(velocity_model_inverted.T, m, clim=clim)
ax2.set(xlabel='Offset [km]', ylabel='Depth [km]', title='Inverted Model')
ax2.set_aspect('auto')
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig23.colorbar(aa)
cbar.ax.set_ylabel('Velocity [km/s]')
fig23.savefig(path_fig+'/model-inverted.png')

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
clim=[-.05, .05]
clim2=[-4e-2, 4e-4]
fig41, ax2 = plt.subplots(figsize=(32,12))
aa=plt.imshow(wavefield_initial, interpolation='nearest', aspect='auto', cmap='seismic', clim=clim,
        extent=[0.0, x_length, t_smp[-1], 0.0])
ax2.set(xlabel='Offset [km]', ylabel='Time [s]', title='Initial wavefield')
ax2.set_aspect('auto')
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig41.colorbar(aa)
cbar.ax.set_ylabel('Amplitude')
fig41.savefig(path_fig+'/wavefield-initial.png')

fig42, ax2 = plt.subplots(figsize=(32,12))
aa=plt.imshow(wavefield_true, interpolation='nearest', aspect='auto', cmap='seismic', clim=clim,
        extent=[0.0, x_length, t_smp[-1], 0.0])
ax2.set(xlabel='Offset [km]', ylabel='Time [s]', title='True wavefield')
ax2.set_aspect('auto')
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig42.colorbar(aa)
cbar.ax.set_ylabel('Amplitude')
fig42.savefig(path_fig+'/wavefield-true.png')

fig43, ax2 = plt.subplots(figsize=(32,12))
aa=plt.imshow(wavefield_inverted, interpolation='nearest', aspect='auto', cmap='seismic', clim=clim,
        extent=[0.0, x_length, t_smp[-1], 0.0])
ax2.set(xlabel='Offset [km]', ylabel='Time [s]', title='Inverted wavefield')
ax2.set_aspect('auto')
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig43.colorbar(aa)
cbar.ax.set_ylabel('Amplitude')
fig43.savefig(path_fig+'/wavefield-inverted.png')

fig44, ax2 = plt.subplots(figsize=(32,12))
aa=plt.imshow((wavefield_true-wavefield_inverted), interpolation='nearest', aspect='auto', cmap='seismic', clim=clim2,
        extent=[0.0, x_length, t_smp[-1], 0.0])
ax2.set(xlabel='Offset [km]', ylabel='Time [s]', title='Difference between wavefields')
ax2.set_aspect('auto')
# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig44.colorbar(aa)
cbar.ax.set_ylabel('Amplitude')
fig44.savefig(path_fig+'/wavefield-difference.png')

############################## Wavefields - receiver #################################
fig5 = plt.figure()
ntr = int(np.shape(wavefield_true)[1]/3 + 1)
tt = (np.arange(np.shape(wavefield_true)[0]) * dt).transpose()
plt.plot(tt, wavefield_true[:, ntr], 'b', label='data-observed')
plt.plot(tt, wavefield_initial[:, ntr], 'g', label='data-initial')
plt.plot(tt, wavefield_inverted[:, ntr], 'r', label='data-inverted')
plt.xlabel('Time [s]')
plt.title('Wavefield-receiver-'+str(ntr))
plt.grid(True)
plt.legend()
fig5.savefig(path_fig+'/wavefield_receiver_'+str(ntr)+'.png')

############################## Wavefields - time #################################
fig6 = plt.figure()
ntr = int(np.shape(wavefield_true)[0]/trange[1] + 1)
aaaa = np.round(ntr * dt, 1)
xx = np.arange(np.shape(wavefield_true)[1]) * m.x.delta
plt.plot(xx, wavefield_true[ntr, :], 'b', label='data-observed')
plt.plot(xx, wavefield_initial[ntr, :], 'g', label='data-initial')
plt.plot(xx, wavefield_inverted[ntr, :], 'r', label='data-inverted')
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

# ############################## verlocity profiles - iterations #################################
# for i in range(trc):
#         fig = plt.figure(figsize=(16,20))
#         ntrc = (trc + i) * delta_x + 1
#         zz = np.arange(np.shape(v0)[0]) * m.z.delta

#         ####################### Here change number of iter $$$$$$$$$$$$$$$$$$$$$$$
#         ##########################################################################
#         plt.plot(vtrue[:, ntrc], zz, 'b', label='True')
#         plt.plot(v1[:, ntrc], zz, 'r', label='iter#1')
#         #plt.plot(v2[:, ntrc], zz, 'c', label='iter#50')
#         plt.plot(v3[:, ntrc], zz, 'y', label='iter#100')
#         #plt.plot(v4[:, ntrc], zz, 'm', label='iter#150')
#         plt.plot(v5[:, ntrc], zz, 'k', label='iter#200')
#         plt.plot(v6[:, ntrc], zz, 'g', label='iter#300')
#         plt.xlabel('velocity [km/s]')
#         plt.ylabel('Depth [km]')
#         plt.title('Traces-'+str(ntrc)+' comparison along z-axis')
#         ax = plt.gca()
#         ax.set_ylim(ax.get_ylim()[::-1])
#         plt.grid(True)
#         plt.legend()
#         fig.savefig(path_fig+'/iter_velocity_file_'+str(ntrc)+'.png')

# ################################################# models - iterations ######################################################
# clim = C.min(),C.max()

# fig2, ax2 = plt.subplots(figsize=(32,12))
# aa=vis.plot(v2.T, m, clim=clim)
# ax2.set(xlabel='Offset [km]', ylabel='Depth [km]', title='Inverted Model')
# ax2.set_aspect('auto')
# # Make a colorbar for the ContourSet returned by the contourf call.
# cbar = fig2.colorbar(aa)
# cbar.ax.set_ylabel('Velocity [km/s]')
# fig2.savefig(path_fig+'/model-iter#100.png')

# fig2, ax2 = plt.subplots(figsize=(32,12))
# aa=vis.plot(v4.T, m, clim=clim)
# ax2.set(xlabel='Offset [km]', ylabel='Depth [km]', title='Inverted Model')
# ax2.set_aspect('auto')
# # Make a colorbar for the ContourSet returned by the contourf call.
# cbar = fig2.colorbar(aa)
# cbar.ax.set_ylabel('Velocity [km/s]')
# fig2.savefig(path_fig+'/model-iter#300.png')

#################### Adjoint source -- L2 ####################################
adj_l2 = wavefield_true - wavefield_initial
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
