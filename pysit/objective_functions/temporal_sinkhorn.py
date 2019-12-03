import time
import numpy as np
from numpy import linalg as la

import scipy
from scipy.stats import norm
from scipy import signal

from pysit.objective_functions.objective_function import ObjectiveFunctionBase
from pysit.util.parallel import ParallelWrapShotNull
from pysit.modeling.temporal_modeling import TemporalModeling

try:
    from scipy.signal import sosfilt
    from scipy.signal import zpk2sos
except ImportError:
    from ._sosfilt import _sosfilt as sosfilt
    from ._sosfilt import _zpk2sos as zpk2sos


__all__ = ['SinkhornDivergence']  # Sinkhorn Divergence

__docformat__ = "restructuredtext en"

ep = 2.2204e-16

class SinkhornDivergence(ObjectiveFunctionBase):
    """ How to compute the parts of the objective you need to do optimization """
    
    def name(self):
        a = 'SinkhornDivergence'
        return a

    def Nr(self):
        return self.nr 

    def Nt(self):
        return self.nt_resampling

    def __init__(self, solver, ot_param, parallel_wrap_shot=ParallelWrapShotNull(), imaging_period=1):
        """imaging_period: Imaging happens every 'imaging_period' timesteps. Use higher numbers to reduce memory consumption at the cost of lower gradient accuracy.
            By assigning this value to the class, it will automatically be used when the gradient function of the temporal objective function is called in an inversion context.
        """
        self.solver = solver
        self.modeling_tools = TemporalModeling(solver)
        self.parallel_wrap_shot = parallel_wrap_shot
        self.imaging_period = int(imaging_period) #Needs to be an integer

        self.sinkhorn_iterations = ot_param['sinkhorn_iterations']
        self.sinkhorn_tolerance = ot_param['sinkhorn_tolerance']
        self.epsilon_maxsmooth = ot_param['epsilon_maxsmooth']
        self.sor = ot_param['successive_over_relaxation']
        self.sign_option = ot_param['sign_option']

        self.epsilon_kl = ot_param['epsilon_kl']
        self.lamb_kl = ot_param['lamb_kl']
        self.x_scale = ot_param['x_scale']
        self.t_scale = ot_param['t_scale']
        self.nt_resampling = ot_param['nt_resampling']
        #self.resample_window = ot_param['resample_window']
        self.nr = ot_param['N_receivers']
        self.filter_op = ot_param['filter_op']
        self.freq_band = ot_param['freq_band']


    # Defintion of functions

    def _maxp(self, d, eps):
        """
        Smooth the max(0, .)
        """
        eps = eps * np.max(d)
        s = 0.5 * (d + np.sqrt(d**2 + eps**2))
        ds = 0.5 * (1 + d/(np.sqrt(d**2 + eps**2))) #* (np.exp(d/np.max(d))-0.8) #(d**2)
        #ds = ds/np.max(ds)
        return s, ds

    def _ot(self, ct, cx, p, q, niter, lamb, epsilon, toler, init_a, init_b):

        sor = self.sor  # Successive over-relaxation

        # utilities
        zp = np.zeros(p.shape)
        op = np.ones(p.shape)
        zq = np.zeros(q.shape)
        oq = np.ones(q.shape)

        le = lamb + epsilon
        pw = lamb/le

        kt = np.exp(-ct/epsilon)
        kx = np.exp(-cx/epsilon)

        ### Without init (a_n,b_n)
        a = op
        b = oq  # a is column and b is line

        ### With init (a_n,b_n)
        # a = init_a
        # b = init_b

        nerr = int(10)
        cvrgce = np.zeros(int(niter/nerr))
        icv = 0  # iteration for errors
        err = 1.0
        ii = 1  # iteration for sinkhorn

        while (ii < niter) and (err > toler):
            ii = ii + 1
            a0 = a
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
        # print('w2_ot converged after %d iterations' %ii)

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

        return dis, grad, cvrgce[0:icv], ar, br #, margin_p, margin_q


    def _mmd(self, ct, cx, p, niter, lamb, epsilon, toler, init_a):

        # utilities
        zp = np.zeros(p.shape)
        op = np.ones(p.shape)

        le = lamb + epsilon
        pw = lamb/le

        kt = np.exp(-ct/epsilon)
        kx = np.exp(-cx/epsilon)

        ### Without init (a_n,b_n)
        a = op

        ### With init (a_n,b_n)
        # a = init_a

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

        # print('w2_mmd converged after %d iterations' %ii)

        uu = -epsilon * np.log((np.dot(kt.dot(a * p), kx))**pw)
        at = -lamb * (np.exp(-uu/lamb)-1)

        gg = np.dot(kt.dot(a * p), kx) * a - np.sum(p)
        a = a * p

        rt = (np.sum(a*(np.dot(kt.dot(a), kx))) - np.sum(p)**2) * epsilon

        dis = 2.0*np.sum(at * p) - rt
        grad = 2.0*(at - epsilon * gg)

        return dis, grad, cvrgce[0:icv], ar


    def _otmmd(self, data_obs, data_cal, t_scale, x_scale, sinkhorn):
        ghk_epsilon = self.epsilon_kl
        ghk_lamb2 = self.lamb_kl
        ghk_niter = self.sinkhorn_iterations
        toler = self.sinkhorn_tolerance
        epmax = self.epsilon_maxsmooth

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
        ps, dps = self._maxp(p, epmax)
        mass_dobs = np.sum(ps)
        ps = ps/mass_dobs

        qs, dqs = self._maxp(q, epmax)
        qs = qs / mass_dobs
        mass_dcal = np.sum(qs)
        
        sinkhorn_init = np.ones([4,np.shape(qs)[0],np.shape(qs)[1]])

        dis, grad, conv, sinkhorn_init[0], sinkhorn_init[1] = self._ot(ct, cx, ps, qs, ghk_niter, ghk_lamb2, ghk_epsilon, toler, sinkhorn[0], sinkhorn[1])
        dis_p, grad_p, conv_p, sinkhorn_init[2] = self._mmd(ct, cx, ps, ghk_niter, ghk_lamb2, ghk_epsilon, toler, sinkhorn[2])
        dis_q, grad_q, conv_q, sinkhorn_init[3] = self._mmd(ct, cx, qs, ghk_niter, ghk_lamb2, ghk_epsilon, toler, sinkhorn[3])

        dis = dis + 0.5*ghk_epsilon*(1.0-mass_dcal)**2 - 0.5*(dis_p + dis_q)
        gradient = grad - 0.5*grad_q - ghk_epsilon*(1.0-mass_dcal)

        adj_src = gradient*dqs/mass_dobs

        return dis, adj_src, sinkhorn_init


    def _residual(self, shot, m0, sinkhorn_p, sinkhorn_n, dWaveOp=None, wavefield=None):
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
        
        dpred = retval['simdata']

        # if shot.background_data is not None:
        #     dpred = retval['simdata'] - shot.background_data
        # else:
        #     dpred = retval['simdata']

        # if shot.receivers.time_window is None:
        #     dpred = dpred
        # else:
        #     dpred = shot.receivers.time_window(self.solver.ts()) * dpred

        # # Filter data
        # n_timesmp = np.shape(dpred)[0]
        # T_max = self.solver.tf
        # filter_op1 = band_pass_filter(n_timesmp, T_max, freq_band=self.freq_band, transit_freq_length=0.5, padding_zeros=True, nl=500, nr=500)
 
        # if self.filter_op is True:
        #     dobs = filter_op1 * shot.receivers.interpolate_data(self.solver.ts())
        #     dpred = filter_op1 * dpred
        # else:
        #     dobs = shot.receivers.interpolate_data(self.solver.ts())

        dobs = shot.receivers.interpolate_data(self.solver.ts())
        shape_dobs = np.shape(dobs)

        # Least square residual
        resid = dobs - dpred  # Residual is the difference between the observed data and predicted data

        # Down-sampling data on time
        dpred_resampled = signal.resample(dpred, self.nt_resampling)
        dobs_resampled = signal.resample(dobs, self.nt_resampling)
       
        ##############################################################################
        if self.sign_option == 'positive':
            dis, adjsrc_resampled, sinkhorn_pos = self._otmmd(dobs_resampled, dpred_resampled, self.t_scale, self.x_scale, sinkhorn_p)
        elif self.sign_option == 'pos+neg':            
            dis_pos, adjsrc_resampled_pos, sinkhorn_pos = self._otmmd(dobs_resampled, dpred_resampled, self.t_scale, self.x_scale, sinkhorn_p)
            dis_neg, adjsrc_resampled_neg, sinkhorn_neg = self._otmmd(-1.0*dobs_resampled, -1.0*dpred_resampled, self.t_scale, self.x_scale, sinkhorn_n)
            
            adjsrc_resampled = adjsrc_resampled_neg - adjsrc_resampled_pos
            dis = dis_pos + dis_neg
        else:
            raise ValueError('Sign option {0} invalid'.format(self.sign_option))
        ##############################################################################
        
        adj_src = signal.resample(adjsrc_resampled, shape_dobs[0]) #, window=self.resample_window)

        # if self.filter_op is True:
        #     adj_src = filter_op1.__adj_mul__(adj_src)

        # If the second derivative info is needed, copy it out
        if dWaveOp is not None:
            dWaveOp[:]  = retval['dWaveOp'][:]
        if wavefield is not None:
            wavefield[:] = retval['wavefield'][:]

        return resid, dis, adj_src, sinkhorn_pos, sinkhorn_neg

    def evaluate(self, shots, m0, sp, sn, **kwargs):
        """ Evaluate the least squares objective function over a list of shots."""

        r_norm2 = 0
        objective_value = 0
        for shot in shots:
            r, dis, adj_src, sinkhorn_pos, sinkhorn_neg = self._residual(shot, m0, sp, sn)
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

        return objective_value  #*self.solver.dt  #, r_norm2*self.solver.dt


    def _gradient_helper(self, shot, m0, sp, sn, ignore_minus=False, ret_pseudo_hess_diag_comp = False, **kwargs):
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
        
        r, dis, adjoint_src, sinkhorn_pos, sinkhorn_neg = self._residual(shot, m0, sp, sn, dWaveOp=dWaveOp, wavefield=wavefield, **kwargs)

        # Perform the migration or F* operation to get the gradient component
        g = self.modeling_tools.migrate_shot(shot, m0, adjoint_src, self.imaging_period, dWaveOp=dWaveOp, wavefield=wavefield)

        if not ignore_minus:
            g = -1*g

        if ret_pseudo_hess_diag_comp:
            return r, g, dis, self._pseudo_hessian_diagonal_component_shot(dWaveOp)
        else:
            return r, g, dis, sinkhorn_pos, sinkhorn_neg, adjoint_src

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

    def compute_gradient(self, shots, m0, sp, sn, aux_info={}, **kwargs):
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
                r, g, dis, sinkhorn_pos, sinkhorn_neg, adjoint_src = self._gradient_helper(shot, m0, sp, sn, ignore_minus=True, **kwargs)
            
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

        return grad, sinkhorn_pos, sinkhorn_neg, adjoint_src, r

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
