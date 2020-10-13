import time
import numpy as np
from numpy import linalg as la

import scipy
from scipy.stats import norm
from scipy import signal
from scipy.interpolate import interp1d

from pysit.objective_functions.objective_function import ObjectiveFunctionBase
from pysit.util.parallel import ParallelWrapShotNull
from pysit.modeling.temporal_modeling import TemporalModeling
import pysit.util.compute_tools as cpt
from pysit.util.compute_tools import get_function
# from scipy.signal import (cheb2ord, cheby2, convolve, get_window, iirfilter, remez)

try:
    from scipy.signal import sosfilt
    from scipy.signal import zpk2sos
except ImportError:
    from ._sosfilt import _sosfilt as sosfilt
    from ._sosfilt import _zpk2sos as zpk2sos

__all__ = ['SinkhornDivergence']  # Sinkhorn Divergence

__docformat__ = "restructuredtext en"

# ep = 2.2204e-16

class SinkhornDivergence(ObjectiveFunctionBase):
    """ How to compute the parts of the objective you need to do optimization """
    
    # Defintion of functions which are used by /pysit/optimization/optimization.py
    def name(self):
        a = 'SinkhornDivergence'
        return a

    def Nr(self):
        if self.nr == 'max':
            return self.solver.mesh.x.n
        else:
            return self.nr 

    def Nt(self, shots):
        if self.dc_factor is not None:
            if (np.shape(shots[0].receivers.data)[0]/self.dc_factor) % 1 == 0:
                nt = int(np.shape(shots[0].receivers.data)[0]/self.dc_factor)
            else:
                nt = int(np.shape(shots[0].receivers.data)[0]/self.dc_factor)+1
            return nt
        else:
            return self.nt_NormSpace


    def __init__(self, solver, ot_param, parallel_wrap_shot=ParallelWrapShotNull(), imaging_period=1):
        """imaging_period: Imaging happens every 'imaging_period' timesteps. Use higher numbers to reduce memory consumption at the cost of lower gradient accuracy.
            By assigning this value to the class, it will automatically be used when the gradient function of the temporal objective function is called in an inversion context.
        """
        self.solver = solver
        self.modeling_tools = TemporalModeling(solver)
        self.parallel_wrap_shot = parallel_wrap_shot
        self.imaging_period = int(imaging_period) #Needs to be an integer

        self.nr = ot_param['n_receivers']
        self.ns = int(self.parallel_wrap_shot.size/2) 

        self.sinkhorn_initialization = ot_param['sinkhorn_initialization']
        self.sinkhorn_iterations = ot_param['sinkhorn_iterations']
        self.sinkhorn_tolerance = ot_param['sinkhorn_tolerance']
        self.sor = ot_param['successive_over_relaxation']
        self.epsilon_kl = ot_param['epsilon_kl']
        self.lamb_kl = ot_param['lamb_kl']
        self.t_scale = ot_param['t_scale']
        self.x_scale = ot_param['x_scale']

        self.data_process_type = ot_param['data_process_type']
        self.trans_func = ot_param['trans_func_type']
        self.trans_factor = ot_param['trans_func_factor']
        self.nt_NormSpace = ot_param['nt_NormSpace']
        self.dc_factor = ot_param['dc_factor']
        self.muteInit = ot_param['muteInit']
        self.padding = ot_param['padding']

        # self.resample_window = ot_param['resample_window']
        self.velocity_bound = ot_param['velocity_bound']
        self.filter_op = ot_param['filter_op']
        self.freq_band = ot_param['freq_band']
        self.noise_factor = ot_param['noise_factor']

    def _ot(self, ct, cx, p, q, niter, lamb, epsilon, toler, a_init=None, b_init=None):

        sor = self.sor  # Successive over-relaxation

        le = lamb + epsilon
        pw = lamb/le

        kt = np.exp(-ct/epsilon)
        kx = np.exp(-cx/epsilon)

        if a_init is None:
            a = np.ones(p.shape)
            b = np.ones(q.shape)
        else:
            a = a_init
            b = b_init

        nerr = int(10)
        cvrgce = np.zeros(int(niter/nerr))
        icv = 0  # iteration for errors
        err = 1.0
        ii = 1  # iteration for sinkhorn

        while (ii < niter) and (err > toler):
            ii = ii + 1
            a0 = a    # a is column and b is line
            b0 = b

            # sinkhorn iterates here
            a = 1.0/(np.dot(kt.dot(b * q), kx)**pw)
            a = a0**(1-sor) * (a**sor)
            b = 1.0/(np.dot(kt.dot(a * p), kx)**pw)
            b = b0**(1-sor) * (b**sor)

            if ii % nerr == 0:
                er = epsilon*(np.log(a)-np.log(a0))
                err = la.norm(er, np.inf)
                cvrgce[icv] = err
                icv = icv + 1

        ar = a
        br = b

        if self.parallel_wrap_shot.use_parallel and (self.parallel_wrap_shot.rank != self.ns):
            []
        else:
            []
            #print('w2_ot converged after %d iterations' %ii)

        vv = -epsilon * np.log((np.dot(kt.dot(a * p), kx))**pw)
        uu = -epsilon * np.log((np.dot(kt.dot(b * q), kx))**pw)

        a = np.exp(uu/epsilon)
        b = np.exp(vv/epsilon)

        at = -lamb * (np.exp(-uu/lamb)-1)
        bt = -lamb * (np.exp(-vv/lamb)-1)

        a = a * p
        gg = np.dot(kt.dot(a), kx) * b - np.sum(p)

        b = b * q
        rt = (np.sum(a*(np.dot(kt.dot(b), kx))) - np.sum(p) * np.sum(q)) * epsilon

        dis = np.sum(at * p + bt * q) - rt
        grad = bt - epsilon * gg

        # # Margins
        # margin_p = (np.dot(kt.dot(a), kx))*b
        # margin_q = (np.dot(kt.dot(b), kx))*a

        if a_init is None:
            return dis, grad, cvrgce[0:icv]
        else:
            return dis, grad, cvrgce[0:icv], ar, br

    def _mmd(self, ct, cx, p, niter, lamb, epsilon, toler, a_init=None):

        le = lamb + epsilon
        pw = lamb/le

        kt = np.exp(-ct/epsilon)
        kx = np.exp(-cx/epsilon)

        if a_init is None:
            a = np.ones(p.shape)
        else:
            a = a_init

        nerr = int(10)
        cvrgce = np.zeros(int(niter/nerr))
        icv = 0  # iteration for errors
        err = 1.0
        ii = 1  # iteration for sinkhorn

        while ((ii < niter) and (err > toler)):

            ii = ii + 1
            a0 = a

            # sinkhorn iterates here
            a = np.sqrt(a/(np.dot(kt.dot(a * p), kx)**pw))

            if ii % nerr == 0:
                er = epsilon*(np.log(a)-np.log(a0))
                err = la.norm(er, np.inf)
                cvrgce[icv] = err
                icv = icv + 1
        ar = a

        if self.parallel_wrap_shot.use_parallel and (self.parallel_wrap_shot.rank != self.ns):
            []
        else:
            []
            #print('w2_mmd converged after %d iterations' %ii)

        uu = -epsilon * np.log((np.dot(kt.dot(a * p), kx))**pw)
        at = -lamb * (np.exp(-uu/lamb)-1)

        gg = np.dot(kt.dot(a * p), kx) * a - np.sum(p)
        a = a * p

        rt = (np.sum(a*(np.dot(kt.dot(a), kx))) - np.sum(p)**2) * epsilon

        dis = 2.0*np.sum(at * p) - rt
        grad = 2.0*(at - epsilon * gg)

        if a_init is None:
            return dis, grad, cvrgce[0:icv]
        else:
            return dis, grad, cvrgce[0:icv], ar


    def _otmmd(self, data_obs, data_cal, t_scale, x_scale, sinkhorn_initial=None):
        ghk_epsilon = self.epsilon_kl
        ghk_lamb2 = self.lamb_kl
        ghk_niter = self.sinkhorn_iterations
        toler = self.sinkhorn_tolerance

        ghk_nt = int(np.shape(data_obs)[0])
        ghk_nx = int(np.shape(data_obs)[1])

        linspt1 = np.linspace(0.0, t_scale, ghk_nt)
        linspt2 = np.linspace(0.0, t_scale, ghk_nt)  # Here we define two linspt for obs and simul separately
        
        linspx1 = np.linspace(0.0, x_scale, ghk_nx)
        linspx2 = np.linspace(0.0, x_scale, ghk_nx)  # Here we define two linspx for obs and simul separately

        t1, t2 = np.meshgrid(linspt1, linspt2)  # T1[Nt,Nt], T2[Nt,Nt] is the meshgrid for loss matrix Ct[Nt, Nt]
        x1, x2 = np.meshgrid(linspx1, linspx2)  # X1[Nx,Nx], X2[Nt,Nx] is the meshgrid for loss matrix Cx[Nx, Nx]

        ct = (t1 - t2)**2
        cx = (x1 - x2)**2

        p = np.asarray(data_obs, dtype=np.float64)
        q = np.asarray(data_cal, dtype=np.float64)
        ct = np.asarray(ct, dtype=np.float64)
        cx = np.asarray(cx, dtype=np.float64)

        # # Normalise the data
        mass_dobs = np.sum(p)
        mass_dcal = np.sum(q)
        p = p/mass_dobs
        q = q/mass_dobs

        #print('mass-processed-dobs = %.8f' %mass_dobs)
        #print('mass-processed-dcal = %.8f' %mass_dcal)

        if sinkhorn_initial is None:
            dis, grad, conv = self._ot(ct, cx, p, q, ghk_niter, ghk_lamb2, ghk_epsilon, toler) #, sinkhorn[0], sinkhorn[1])
            dis_p, grad_p, conv_p = self._mmd(ct, cx, p, ghk_niter, ghk_lamb2, ghk_epsilon, toler) #, sinkhorn[2])
            dis_q, grad_q, conv_q = self._mmd(ct, cx, q, ghk_niter, ghk_lamb2, ghk_epsilon, toler) #, sinkhorn[3])
        else:
            sinkhorn_output = np.zeros_like(np.copy(sinkhorn_initial))
            dis, grad, conv, sinkhorn_output[0], sinkhorn_output[1] = self._ot(ct, cx, p, q, ghk_niter, ghk_lamb2, ghk_epsilon, toler, sinkhorn_initial[0], sinkhorn_initial[1])
            dis_p, grad_p, conv_p, sinkhorn_output[2] = self._mmd(ct, cx, p, ghk_niter, ghk_lamb2, ghk_epsilon, toler, sinkhorn_initial[2])
            dis_q, grad_q, conv_q, sinkhorn_output[3] = self._mmd(ct, cx, q, ghk_niter, ghk_lamb2, ghk_epsilon, toler, sinkhorn_initial[3])

        dis = dis + 0.5*ghk_epsilon*(1.0-mass_dcal)**2 - 0.5*(dis_p + dis_q)
        gradient = grad - 0.5*grad_q - ghk_epsilon*(1.0-mass_dcal)

        adj_src = gradient/mass_dobs
        # adj_src = gradient*dqs/mass_dobs

        if sinkhorn_initial is None:
            return dis, adj_src
        else:
            return dis, adj_src, sinkhorn_output


    def _data_processing(self, shot, retval, ntn, dp_type=None, iter_dp=None):

        #if shot.background_data is not None:
        #    dpred = retval['simdata'] - shot.background_data
        #else:
        #    dpred = retval['simdata']

        #if shot.receivers.time_window is not None:
        #    dpred = shot.receivers.time_window(self.solver.ts()) * dpred
        dpred = retval['simdata']    
        dobs = shot.receivers.data

        nto = int(np.shape(dobs)[0])
        ntc = int(np.shape(dpred)[0])
        nxc = int(np.shape(dpred)[1])

        t0, tf = self.solver.trange
        to = np.arange(nto)*(tf-t0) / float(nto-1)
        tc = np.arange(ntc)*(tf-t0) / float(ntc-1)
        tn = np.arange(ntn)*(tf-t0) / float(ntn-1)

        dobs_interp = shot.receivers.interpolate_data(self.solver.ts())

        # Least square residual
        resid = dobs_interp - dpred  # Residual is the difference between the observed data and predicted data

        ###################### Mute first arrivals #################
        if self.muteInit is not None:

            if iter_dp != 0:
                dinit = self.muteInit[self.parallel_wrap_shot.rank]
                nti = int(np.shape(dinit)[0])
                ti = np.arange(nti)*(tf-t0) / float(nti-1)

                f = interp1d(ti, dinit, axis=0, kind='linear', copy=False, bounds_error=False, fill_value=0.0)
                dinit_obs = f(to)
                dinit_pred = f(tc)

                dobs = dobs - dinit_obs
                dpred = dpred - dinit_pred
                dobs_interp = dobs_interp - dinit_pred

        ###################### Noise  #################
        if self.noise_factor is not None:
            for i in range(nxc):
                noise = np.zeros([nto,1])
                mu, sigma = 0, np.max(dobs[:,i])*self.noise_factor
                noise = np.random.normal(mu, sigma, nto)
                dobs[:,i] = dobs[:,i] + noise

                noise_interp = np.zeros([ntc,1])
                noise_interp = np.random.normal(mu, sigma, ntc)
                dobs_interp[:,i] = dobs_interp[:,i] + noise_interp

        # ###################### Filtering  ################# 
        # if self.filter_op is True:
        #     n_timesmp = np.shape(dpred)[0]
        #     T_max = self.solver.tf
        #     filter_op1 = band_pass_filter(n_timesmp, T_max, freq_band=self.freq_band, transit_freq_length=0.5, padding_zeros=True, nl=500, nr=500)

        #     dobs = filter_op1.__adj_mul__(filter_op1 * dobs)
        #     dpred = filter_op1.__adj_mul__(filter_op1 * dpred)

        ###################### Zeros-padding #################
        if self.padding is not None:
            nl, nr = self.padding
            win = signal.tukey(10)

            dobs_pad = np.zeros([nto+nl+nr, nx])
            dobs_interp_pad = np.zeros([ntc+nl+nr, nx])
            dpred_pad = np.zeros([ntc+nl+nr, nx])

            dobs_pad_win = np.zeros([nto+nl+nr, nx])
            dobs_interp_pad_win = np.zeros([ntc+nl+nr, nx])
            dpred_pad_win = np.zeros([ntc+nl+nr, nx])

            for i in range(nxc):
                dobs_pad[:,i] = cpt.padding_zeros_fun(dobs[:,i], nto, nl, nr)
                dobs_interp_pad[:,i] = cpt.padding_zeros_fun(dobs_interp[:,i], ntc, nl, nr)
                dpred_pad[:,i] = cpt.padding_zeros_fun(dpred[:,i], ntc, nl, nr)

                dobs_pad_win[:,i] = signal.convolve(dobs_pad[:,i], win, mode='same') / sum(win)
                dobs_interp_pad_win[:,i] = signal.convolve(dobs_interp_pad[:,i], win, mode='same') / sum(win)
                dpred_pad_win[:,i] = signal.convolve(dpred_pad[:,i], win, mode='same') / sum(win)

            # ntop = np.shape(dobs_pad_win)[0]
            # ntcp = np.shape(dpred_pad_win)[0]

            # dtop = (tf-t0) / float(ntop-1)
            # dtcp = (tf-t0) / float(ntcp-1)

            # t_smp_ntop = np.linspace(t0-dtop*(nl+nr)), tf+dtop*(nl+nr), ntop)
            # t_smp_ntpc = np.linspace(t0-dtcp*(nl+nr)), tf+dtcp*(nl+nr), ntcp)


        if dp_type == 'IDsT':

            ###################### Down-Sampling ##################
            dpred_ds = signal.resample(dpred, self.nt_NormSpace)
            dobs_ds = signal.resample(dobs_interp, self.nt_NormSpace)

            ###################### Transforming ##################
            tpvs, tpvs_grad = get_function(self.trans_func)
            dobs_pv = tpvs(dobs_ds, factor=self.trans_factor)
            dpred_pv = tpvs(dpred_ds, factor=self.trans_factor)
            dpred_pv_grad = tpvs_grad(dpred_ds, factor=self.trans_factor) 

            return dobs_pv, dpred_pv, dpred_pv_grad, resid, ntc  

        elif dp_type == 'DsT':

            ###################### Down-Sampling ##################
            dpred_ds = signal.resample(dpred, self.nt_NormSpace)
            dobs_ds = signal.resample(dobs, self.nt_NormSpace)

            ###################### Transforming ##################
            tpvs, tpvs_grad = get_function(self.trans_func)
            dobs_pv = tpvs(dobs_ds, factor=self.trans_factor)
            dpred_pv = tpvs(dpred_ds, factor=self.trans_factor)
            dpred_pv_grad = tpvs_grad(dpred_ds, factor=self.trans_factor)

            return dobs_pv, dpred_pv, dpred_pv_grad, resid, ntc
        
        elif dp_type == 'IDcT':

            ###################### Interpolating ###############
            f = interp1d(tc, dpred, axis=0, kind='linear', copy=False, bounds_error=False, fill_value=0.0)
            dpred_interp = f(to)

            ###################### Decimating ##################
            dobs_ds = signal.decimate(dobs, self.dc_factor, ftype='fir', axis=0, zero_phase=True)
            dpred_ds = signal.decimate(dpred_interp, self.dc_factor, ftype='fir', axis=0, zero_phase=True)

            ###################### Transforming ##################
            tpvs, tpvs_grad = get_function(self.trans_func)
            dobs_pv = tpvs(dobs_ds, factor=self.trans_factor)
            dpred_pv = tpvs(dpred_ds, factor=self.trans_factor)
            dpred_pv_grad = tpvs_grad(dpred_ds, factor=self.trans_factor)

            return dobs_pv, dpred_pv, dpred_pv_grad, resid, ntc

        elif dp_type == 'TDs':

            ####################### Transforming ##################
            tpvs, tpvs_grad = get_function(self.trans_func)
            dobs_pv = tpvs(dobs, factor=self.trans_factor)
            dpred_pv = tpvs(dpred, factor=self.trans_factor)
            dpred_pv_grad = tpvs_grad(dpred, factor=self.trans_factor)

            ###################### Down-Sampling  #################
            dobs_pd = np.zeros([2,ntn,nxc], dtype=float)
            dpred_pd = np.zeros([2,ntn,nxc], dtype=float)
            dpred_pd_grad = np.zeros([2,ntn,nxc], dtype=float)

            dobs_pd[0][:][:] = signal.resample(dobs_pv[0], self.nt_NormSpace)
            dobs_pd[1][:][:] = signal.resample(dobs_pv[1], self.nt_NormSpace)
            dpred_pd[0][:][:] = signal.resample(dpred_pv[0], self.nt_NormSpace) 
            dpred_pd[1][:][:] = signal.resample(dpred_pv[1], self.nt_NormSpace)   
            dpred_pd_grad[0][:][:] = signal.resample(dpred_pv_grad[0], self.nt_NormSpace)   
            dpred_pd_grad[1][:][:] = signal.resample(dpred_pv_grad[1], self.nt_NormSpace)

            # print('test np.maximum(a, 0.0)')
            # print(np.min(dobs_pd))
            # print(np.min(dpred_pd))
            dobs_pd = np.maximum(dobs_pd, 0.0)
            dpred_pd = np.maximum(dpred_pd, 0.0)
            # print(np.min(dobs_pd))
            # print(np.min(dpred_pd))

            return dobs_pd, dpred_pd, dpred_pd_grad, resid, ntc

        elif dp_type == 'ITDc':

            ###################### Interpolating ###############
            f = interp1d(tc, dpred, axis=0, kind='linear', copy=False, bounds_error=False, fill_value=0.0)
            dpred_interp = f(to)

            ####################### Transforming ##################
            tpvs, tpvs_grad = get_function(self.trans_func)
            dobs_pv = tpvs(dobs, factor=self.trans_factor)
            dpred_pv = tpvs(dpred_interp, factor=self.trans_factor)
            dpred_pv_grad = tpvs_grad(dpred_interp, factor=self.trans_factor)

            if (nto/self.dc_factor) % 1 == 0:
                ntn_dc = int(nto/self.dc_factor)
            else:
                ntn_dc = int(nto/self.dc_factor)+1
            ###################### Decimating ##################
            dobs_pd = np.zeros([2,ntn_dc,nxc], dtype=float)
            dpred_pd = np.zeros([2,ntn_dc,nxc], dtype=float)
            dpred_pd_grad = np.zeros([2,ntn_dc,nxc], dtype=float)

            dobs_pd[0][:][:] = signal.decimate(dobs_pv[0], self.dc_factor, ftype='fir', axis=0, zero_phase=True)
            dobs_pd[1][:][:] = signal.decimate(dobs_pv[1], self.dc_factor, ftype='fir', axis=0, zero_phase=True)
            dpred_pd[0][:][:] = signal.decimate(dpred_pv[0], self.dc_factor, ftype='fir', axis=0, zero_phase=True)
            dpred_pd[1][:][:] = signal.decimate(dpred_pv[1], self.dc_factor, ftype='fir', axis=0, zero_phase=True)
            dpred_pd_grad[0][:][:] = signal.decimate(dpred_pv_grad[0], self.dc_factor, ftype='fir', axis=0, zero_phase=True)  
            dpred_pd_grad[1][:][:] = signal.decimate(dpred_pv_grad[1], self.dc_factor, ftype='fir', axis=0, zero_phase=True)

            # print('test np.maximum(a, 0.0)')
            # print(np.min(dobs_pd))
            # print(np.min(dpred_pd))
            dobs_pd = np.maximum(dobs_pd, 0.0)
            dpred_pd = np.maximum(dpred_pd, 0.0)
            # print(np.min(dobs_pd))
            # print(np.min(dpred_pd))

            return dobs_pd, dpred_pd, dpred_pd_grad, resid, ntc

        elif dp_type == 'TI':
            ###################### Transforming ##################
            tpvs, tpvs_grad = get_function(self.trans_func)
            dobs_pv = tpvs(dobs, factor=self.trans_factor)
            dpred_pv = tpvs(dpred, factor=self.trans_factor)
            dpred_pv_grad = tpvs_grad(dpred, factor=self.trans_factor)   

            ###################### Interpolating  #################
            dobs_pd = np.zeros([2,ntn,nxc], dtype=float)
            dpred_pd = np.zeros([2,ntn,nxc], dtype=float)
            dpred_pd_grad = np.zeros([2,ntn,nxc], dtype=float)

            fo0 = interp1d(to, dobs_pv[0], axis=0, kind='linear', copy=False, bounds_error=False, fill_value=0.0)
            fo1 = interp1d(to, dobs_pv[1], axis=0, kind='linear', copy=False, bounds_error=False, fill_value=0.0)
            fc0 = interp1d(tc, dpred_pv[0], axis=0, kind='linear', copy=False, bounds_error=False, fill_value=0.0)
            fc1 = interp1d(tc, dpred_pv[1], axis=0, kind='linear', copy=False, bounds_error=False, fill_value=0.0)
            fcg0 = interp1d(tc, dpred_pv_grad[0], axis=0, kind='linear', copy=False, bounds_error=False, fill_value=0.0)
            fcg1 = interp1d(tc, dpred_pv_grad[1], axis=0, kind='linear', copy=False, bounds_error=False, fill_value=0.0)

            dobs_pd[0][:][:] = fo0(tn)
            dobs_pd[1][:][:] = fo1(tn)
            dpred_pd[0][:][:] = fc0(tn)
            dpred_pd[1][:][:] = fc1(tn)
            dpred_pd_grad[0][:][:] = fcg0(tn)
            dpred_pd_grad[1][:][:] = fcg1(tn)

            return dobs_pd, dpred_pd, dpred_pd_grad, resid, ntc

        else:
            raise Exception("None data processing type is defined")


    def _residual(self, shot, m0, sinkhorn_init=None, iter_rs=None, dWaveOp=None, wavefield=None):
        """Computes residual in the usual sense.

        Parameters
        ----------
        shot : pysit.Shot
            Shot for which to compute the residual.
        dWaveOp : list of ndarray (optional)
            An empty list for returning the derivative term required for
            computing the imaging condition.

        """

        # If we will use the second derivative info later (and this is usually
        # the case in inversion), tell the solver to store that information, in
        # addition to the solution as it would be observed by the receivers in
        # this shot (aka, the simdata).
        rp = ['simdata']
        if dWaveOp is not None:
            rp.append('dWaveOp')
        # If we are dealing with variable density, we want the wavefield returned as well.
        if wavefield is not None:
            rp.append('wavefield')

        # Run the forward modeling step
        retval = self.modeling_tools.forward_model(shot, m0, self.imaging_period, return_parameters=rp)
        # Compute the residual vector by interpolating the measured data to the
        # timesteps used in the previous forward modeling stage.
        # resid = map(lambda x,y: x.interpolate_data(self.solver.ts())-y, shot.gather(), retval['simdata'])

        dobs_pv, dpred_pv, dpred_pv_grad, resid, ntc = self._data_processing(shot, retval, self.nt_NormSpace, dp_type=self.data_process_type, iter_dp=iter_rs)

        # for elem_tpvs in tpvs:
        distance = 0.0
        adjsrc_ds = np.zeros([np.shape(dobs_pv)[1], np.shape(dobs_pv)[2]])
        if sinkhorn_init is None:
            for i in range(len(dobs_pv)):
                dis, adj = self._otmmd(dobs_pv[i], dpred_pv[i], self.t_scale, self.x_scale)
                adj = adj*dpred_pv_grad[i]
                distance += dis
                adjsrc_ds -= adj
        else:
            sinkhorn_output = np.zeros_like(np.copy(sinkhorn_init))
            for i in range(len(dobs_pv)):
                dis, adj, sinkhorn_output[i] = self._otmmd(dobs_pv[i], dpred_pv[i], self.t_scale, self.x_scale, sinkhorn_init[i])
                adj = adj*dpred_pv_grad[i]
                distance += dis
                adjsrc_ds -= adj
   
        adj_src = signal.resample(adjsrc_ds, ntc) #, window=self.resample_window)
        
        # If the second derivative info is needed, copy it out
        if dWaveOp is not None:
            dWaveOp[:]  = retval['dWaveOp'][:]
        if wavefield is not None:
            wavefield[:] = retval['wavefield'][:]

        if sinkhorn_init is None:
            return resid, distance, adj_src 
        else:
            return resid, distance, adj_src, sinkhorn_output

    def evaluate(self, shots, m0, sinkhorn_init=None, **kwargs):
        """ Evaluate the least squares objective function over a list of shots."""

        r_norm2 = 0
        objective_value = 0
        for shot in shots:
            if sinkhorn_init is None:
                r, dis, adj_src = self._residual(shot, m0)
            else:
                r, dis, adj_src, sinkhorn_output = self._residual(shot, m0, sinkhorn_init)
    
            r_norm2 += np.linalg.norm(r)**2
            objective_value += dis

        # sum-reduce and communicate result
        if self.parallel_wrap_shot.use_parallel:
            # Allreduce wants an array, so we give it a 0-D array
            new_r_norm2 = np.array(0.0)
            self.parallel_wrap_shot.comm.Allreduce(np.array(r_norm2), new_r_norm2)
            r_norm2 = new_r_norm2[()] # goofy way to access 0-D array element

            new_objective_value = np.array(0.0)
            self.parallel_wrap_shot.comm.Allreduce(np.array(objective_value), new_objective_value)
            objective_value = new_objective_value[()] # goofy way to access 0-D array element

        if sinkhorn_init is None:
            return objective_value  #*self.solver.dt  #, r_norm2*self.solver.dt
        else:
            return objective_value, sinkhorn_output

    def _gradient_helper(self, shot, m0, sinkhorn_init=None, iter_gh=None, ignore_minus=False, ret_pseudo_hess_diag_comp = False, **kwargs):
        """Helper function for computing the component of the gradient due to a
        single shot.

        Computes F*_s(d - scriptF_s[u]), in our notation.

        Parameters
        ----------
        shot : pysit.Shot
            Shot for which to compute the residual.

        """

        # Compute the residual vector and its norm
        dWaveOp=[]

        # If this is true, then we are dealing with variable density. In this case, we want our forward solve
        # To also return the wavefield, because we need to take gradients of the wavefield in the adjoint model
        # Step to calculate the gradient of our objective in terms of m2 (ie. 1/rho)
        if hasattr(m0, 'kappa') and hasattr(m0,'rho'):
            wavefield=[]
        else:
            wavefield=None
        
        if sinkhorn_init is None:
            r, dis, adjoint_src = self._residual(shot, m0, iter_rs=iter_gh, dWaveOp=dWaveOp, wavefield=wavefield, **kwargs)
        else:
            r, dis, adjoint_src, sinkhorn_output = self._residual(shot, m0, sinkhorn_init, iter_rs=iter_gh, dWaveOp=dWaveOp, wavefield=wavefield, **kwargs)

        # Perform the migration or F* operation to get the gradient component
        g = self.modeling_tools.migrate_shot(shot, m0, adjoint_src, self.imaging_period, dWaveOp=dWaveOp, wavefield=wavefield)

        if not ignore_minus:
            g = -1*g

        if ret_pseudo_hess_diag_comp:
            return r, g, dis, self._pseudo_hessian_diagonal_component_shot(dWaveOp)
        else:
            if sinkhorn_init is None:
                return r, g, dis, adjoint_src
            else:
                return r, g, dis, adjoint_src, sinkhorn_output

    def _pseudo_hessian_diagonal_component_shot(self, dWaveOp):
        #Shin 2001: "Improved amplitude preservation for prestack depth migration by inverse scattering theory". 
        #Basic illumination compensation. In here we compute the diagonal. It is not perfect, it does not include receiver coverage for instance.
        #Currently only implemented for temporal modeling. Although very easy for frequency modeling as well. -> np.real(omega^4*wavefield * np.conj(wavefield)) -> np.real(dWaveOp*np.conj(dWaveOp))
        
        mesh = self.solver.mesh
          
        tt = time.time()
        pseudo_hessian_diag_contrib = np.zeros(mesh.unpad_array(dWaveOp[0], copy=True).shape)
        for i in range(len(dWaveOp)):                          #Since dWaveOp is a list I cannot use a single numpy command but I need to loop over timesteps. May have been nicer if dWaveOp had been implemented as a single large ndarray I think
            unpadded_dWaveOp_i = mesh.unpad_array(dWaveOp[i])   #This will modify dWaveOp[i] ! But that should be okay as it will not be used anymore.
            pseudo_hessian_diag_contrib += unpadded_dWaveOp_i*unpadded_dWaveOp_i

        pseudo_hessian_diag_contrib *= self.imaging_period #Compensate for doing fewer summations at higher imaging_period

        print("Time elapsed when computing pseudo hessian diagonal contribution shot: %e"%(time.time() - tt))

        return pseudo_hessian_diag_contrib

    def compute_gradient(self, shots, m0, sinkhorn_init=None, iter_nb=None, aux_info={}, **kwargs):
        """Compute the gradient for a set of shots.

        Computes the gradient as
            -F*(d - scriptF[m0]) = -sum(F*_s(d - scriptF_s[m0])) for s in shots

        Parameters
        ----------
        shots : list of pysit.Shot
            List of Shots for which to compute the gradient.
        m0 : ModelParameters
            The base point about which to compute the gradient
        """


        # compute the portion of the gradient due to each shot
        grad = m0.perturbation()
        r_norm2 = 0.0
        objective_value = 0
        pseudo_h_diag = np.zeros(m0.asarray().shape)
        for shot in shots:
            if ('pseudo_hess_diag' in aux_info) and aux_info['pseudo_hess_diag'][0]:
                r, g, dis, h = self._gradient_helper(shot, m0, ignore_minus=True, ret_pseudo_hess_diag_comp = True, **kwargs)
                pseudo_h_diag += h 
            else:
                if sinkhorn_init is None:
                    r, g, dis, adjoint_src = self._gradient_helper(shot, m0, iter_gh=iter_nb, ignore_minus=True, **kwargs)
                else:
                    r, g, dis, adjoint_src, sinkhorn_output = self._gradient_helper(shot, m0, sinkhorn_init, iter_gh=iter_nb, ignore_minus=True, **kwargs)
            
            grad -= g # handle the minus 1 in the definition of the gradient of this objective
            r_norm2 += np.linalg.norm(r)**2
            objective_value += dis

        # sum-reduce and communicate result
        if self.parallel_wrap_shot.use_parallel:
            # Allreduce wants an array, so we give it a 0-D array
            new_r_norm2 = np.array(0.0)
            self.parallel_wrap_shot.comm.Allreduce(np.array(r_norm2), new_r_norm2)
            r_norm2 = new_r_norm2[()] # goofy way to access 0-D array element

            new_objective_value = np.array(0.0)
            self.parallel_wrap_shot.comm.Allreduce(np.array(objective_value), new_objective_value)
            objective_value = new_objective_value[()] # goofy way to access 0-D array element

            ngrad = np.zeros_like(grad.asarray())
            self.parallel_wrap_shot.comm.Allreduce(grad.asarray(), ngrad)
            grad=m0.perturbation(data=ngrad)
            
            if ('pseudo_hess_diag' in aux_info) and aux_info['pseudo_hess_diag'][0]:
                pseudo_h_diag_temp = np.zeros(pseudo_h_diag.shape)
                self.parallel_wrap_shot.comm.Allreduce(pseudo_h_diag, pseudo_h_diag_temp)
                pseudo_h_diag = pseudo_h_diag_temp 

        # account for the measure in the integral over time
        r_norm2 *= self.solver.dt
        # objective_value *=self.solver.dt
        pseudo_h_diag *= self.solver.dt #The gradient is implemented as a time integral in TemporalModeling.adjoint_model(). I think the pseudo Hessian (F*F in notation Shin) also represents a time integral. So multiply with dt as well to be consistent.

        # store any auxiliary info that is requested
        if ('residual_norm' in aux_info) and aux_info['residual_norm'][0]:
            aux_info['residual_norm'] = (True, np.sqrt(r_norm2))
        if ('objective_value' in aux_info) and aux_info['objective_value'][0]:
            aux_info['objective_value'] = (True, objective_value)
        if ('pseudo_hess_diag' in aux_info) and aux_info['pseudo_hess_diag'][0]:
            aux_info['pseudo_hess_diag'] = (True, pseudo_h_diag)

        if sinkhorn_init is None:
            return grad, adjoint_src, r
        else:
            return grad, adjoint_src, r, sinkhorn_output

    def apply_hessian(self, shots, m0, m1, hessian_mode='approximate', levenberg_mu=0.0, *args, **kwargs):

        modes = ['approximate', 'full', 'levenberg']
        if hessian_mode not in modes:
            raise ValueError("Invalid Hessian mode.  Valid options for applying hessian are {0}".format(modes))

        result = m0.perturbation()

        if hessian_mode in ['approximate', 'levenberg']:
            for shot in shots:
                # Run the forward modeling step
                retval = self.modeling_tools.forward_model(shot, m0, return_parameters=['dWaveOp'])
                dWaveOp0 = retval['dWaveOp']

                linear_retval = self.modeling_tools.linear_forward_model(shot, m0, m1, return_parameters=['simdata'], dWaveOp0=dWaveOp0)

                d1 = linear_retval['simdata'] # data from F applied to m1
                result += self.modeling_tools.migrate_shot(shot, m0, d1, dWaveOp=dWaveOp0)
                linear_retval = self.modeling_tools.linear_forward_model(shot, m0, m1, return_parameters=['simdata', 'dWaveOp1'], dWaveOp0=dWaveOp0)
                d1 = linear_retval['simdata']
                dWaveOp1 = linear_retval['dWaveOp1']

                # <q, u1tt>, first adjointy bit
                dWaveOpAdj1=[]
                res1 = self.modeling_tools.migrate_shot( shot, m0, r0, dWaveOp=dWaveOp1, dWaveOpAdj=dWaveOpAdj1)
                result += res1

                # <p, u0tt>
                res2 = self.modeling_tools.migrate_shot(shot, m0, d1, operand_dWaveOpAdj=dWaveOpAdj1, operand_model=m1, dWaveOp=dWaveOp0)
                result += res2

        # sum-reduce and communicate result
        if self.parallel_wrap_shot.use_parallel:

            nresult = np.zeros_like(result.asarray())
            self.parallel_wrap_shot.comm.Allreduce(result.asarray(), nresult)
            result = m0.perturbation(data=nresult)

        # Note, AFTER the application has been done in parallel do this.
        if hessian_mode == 'levenberg':
            result += levenberg_mu*m1

        return result
