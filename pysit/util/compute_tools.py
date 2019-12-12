import numpy as np
import copy as copy

import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import hilbert

import obspy.io.segy.core as segy

__all__ = ['odn2grid', 'odn2grid_data_2D_time', 'odn2grid_data_3D_time',
           'odn2grid_data_2D_freq', 'odn2grid_data_3D_freq', 'low_pass_filter',
           'high_pass_filter', 'band_pass_filter', 'correlate_fun', 'optimal_transport_fwi', 
           'padding_zeros_fun', 'un_padding_zeros_fun', 'padding_zeros_op', 'envelope_fun',
           'opSmooth1D', 'opSmooth2D', 'get_function']

ep = 2.2204e-16

def get_function(func_id):
    """
    Return function from its id

    Parameter
    ---------
    func_id : str
        function id

    Return
    ------
    Callable object
    """
    switcher = {'id'                : identity,
                'id2'               : identity2,
                'exp'               : exponential,
                'square'            : square,
                'smooth_max'        : smoothMax,
               }

    switcher_gradient = {'id'           : identity_gradient,
                         'id2'          : identity2_gradient,
                         'exp'          : exponential_gradient,
                         'square'       : square_gradient,
                         'smooth_max'   : smoothMax_gradient,
                        }

    return switcher.get(func_id, None), switcher_gradient.get(func_id, None)

def identity(x):
    nt=np.shape(x)[0]
    nx=np.shape(x)[1]
    a = np.zeros([nt,nx], dtype=float), 

    a[0][:][:] = np.asarray(x, dtype=float)

    return a

def identity_gradient(x):
    nt=np.shape(x)[0]
    nx=np.shape(x)[1]
    a = np.zeros([nt,nx], dtype=float), 

    a[0][:][:] = np.ones_like(np.copy(x), dtype=float)
    return a

def identity2(x):
    a = np.asarray(x, dtype=float)
    return a, a

def identity2_gradient(x):
    a = np.ones_like(np.copy(x), dtype=float)
    return a, a

def exponential(x):
    nt=np.shape(x)[0]
    nx=np.shape(x)[1]
    a = np.zeros([nt,nx], dtype=float), 

    a[0][:][:] = np.exp(x)
    return a

def exponential_gradient(x):
    nt=np.shape(x)[0]
    nx=np.shape(x)[1]
    a = np.zeros([nt,nx], dtype=float), 

    a[0][:][:] = np.exp(x)
    return a

def square(x):
    nt=np.shape(x)[0]
    nx=np.shape(x)[1]
    a = np.zeros([nt,nx], dtype=float), 

    a[0][:][:] = np.asarray(0.5*(x**2), dtype=float)

    return a

def square_gradient(x):
    nt=np.shape(x)[0]
    nx=np.shape(x)[1]
    a = np.zeros([nt,nx], dtype=float), 

    a[0][:][:] = np.asarray(x, dtype=float)
    return a

def smoothMax(x):
    x_pos = 0.5 * (x + np.sqrt(x**2 + (ep * np.max(x))**2))
    x_neg = 0.5 * (-x + np.sqrt(x**2 + (ep * np.max(-x))**2))
    return x_pos, x_neg

def smoothMax_gradient(x):
    dx_pos = 0.5 * (1 + x/(np.sqrt(x**2 + (ep * np.max(x))**2))) #* (np.exp(d/np.max(d))-0.8) #(d**2)
    dx_neg = 0.5 * (1 - x/(np.sqrt(x**2 + (ep * np.max(-x))**2))) #* (np.exp(d/np.max(d))-0.8) #(d**2)
    return dx_pos, dx_neg

###############################################
def odn2grid(o, d, n):
    output = dict()
    for i in range(len(o)):
        tmp = np.array(range(0, n[i])).astype(float)
        output[i] = o[i] + tmp*d[i]

    return output

def odn2grid_data_2D_time(o, d, n):
    output = odn2grid(o, d, n)
    data_time = output[0]
    data_xrec = output[1]
    data_zrec = output[2]
    data_xsrc = output[3]
    data_zsrc = output[4]

    return data_time, data_xrec, data_zrec, data_xsrc, data_zsrc


def odn2grid_data_3D_time(o, d, n):
    output = odn2grid(o, d, n)
    data_time = output[0]
    data_xrec = output[1]
    data_yrec = output[2]
    data_zrec = output[3]
    data_xsrc = output[4]
    data_ysrc = output[5]
    data_zsrc = output[6]

    return data_time, data_xrec, data_yrec, data_zrec, data_xsrc, data_ysrc, data_zsrc


def odn2grid_data_2D_freq(o, d, n):
    output = odn2grid(o, d, n)
    data_xrec = output[0]
    data_zrec = output[1]
    data_xsrc = output[2]
    data_zsrc = output[3]
    data_freq = output[4]

    return data_xrec, data_zrec, data_xsrc, data_zsrc, data_freq


def odn2grid_data_3D_freq(o, d, n):
    output = odn2grid(o, d, n)
    data_xrec = output[0]
    data_yrec = output[1]
    data_zrec = output[2]
    data_xsrc = output[3]
    data_ysrc = output[4]
    data_zsrc = output[5]
    data_freq = output[6]

    return data_xrec, data_yrec, data_zrec, data_xsrc, data_ysrc, data_zsrc, data_freq

class opSmooth1D(object):
    '''This is a operator to smooth a 1D data
        
    '''

    def __init__(self, n, n_conv, window_len=3, axis=0, window='flat'):
        '''
            Input:
            n: number of data points
            n_conv : number of convolution applied
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman',
             flat window will produce a moving average smoothing.
        '''

        self.nsmp = n
        self.window_len = window_len
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        else:
            self.window = window 
        self.shape = [n,n]
        self.axis = axis
        self.n_conv = n_conv

    def _apply_smooth1d(self, x):
        if x.size != self.nsmp:
            raise ValueError("The length of the input vector does not equal to nsmp of the smoothing operator")
        if x.size < self.window_len:
            raise ValueError("Input vector needs to be bigger than window size.")

        s = np.r_[x[self.window_len-1:0:-1], x, x[-2:-self.window_len-1:-1]]

        if self.window == 'flat':  # moving average
            w = np.ones(self.window_len, 'd')
        else:
            w = eval('np.'+self.window+'(self.window_len)')

        y = np.convolve(w/w.sum(), s, mode='valid')

        return y[((self.window_len+1)//2-1):len(y)-((self.window_len-1)//2)] 

    def __mul__(self, x):
        x_shape = np.shape(x)
        y = copy.deepcopy(x)

        if len(x_shape) == 1:
            if x_shape[0] != self.shape[1]:
                raise ValueError('The size of input x should be equal to the shape[1] of the high pass filter object')

            else:
                for i in range(0, self.n_conv):
                    y = self._apply_smooth1d(y)


        else:
            if x_shape[self.axis] != self.shape[1]:
                raise ValueError(
                    "The length of input x's operating axis should be equal to the shape[1] of the high pass filter object")
            else:
                for i in range(0, self.n_conv):
                    y = np.apply_along_axis(self._apply_smooth1d, self.axis, y)

        return y

class opSmooth2D(object):
    '''This is a operator to smooth a 2D Image
        
    '''

    def __init__(self, n, n_conv, window_len=[3, 3], window='hanning', axis=0):
        '''
            Input:
            n: number of data points
            n_conv : number of convolution applied on each dimension
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman',
             flat window will produce a moving average smoothing.
        '''

        self.nsmp = np.prod(n)
        self.window_len = window_len
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError(
                "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        else:
            self.window = window
        self.shape = [self.nsmp, self.nsmp]
        self.target_size = n
        self.S0 = opSmooth1D(n[0], n_conv[0], window_len=window_len[0], axis=0, window=window)
        self.S1 = opSmooth1D(n[1], n_conv[1], window_len=window_len[1], axis=1, window=window)
        self.axis = axis
        self.n_conv = n_conv

    def __mul__(self, x):
        x_shape = np.shape(x)

        if len(x_shape) == 1:
            if x_shape[0] != self.shape[1]:
                raise ValueError(
                    'The size of input x should be equal to the shape[1] of the high pass filter object')

            else:
               y = self._apply_1d(x)

        else:
            if x_shape[self.axis] != self.shape[1]:
                raise ValueError(
                    "The length of input x's operating axis should be equal to the shape[1] of the high pass filter object")
            else:
                y = np.apply_along_axis(self._apply_1d, self.axis, x)

        return y

    def _apply_1d(self, x):
         x1 = np.reshape(x, self.target_size)
         y = self.S1*(self.S0 * x1)
         y = np.reshape(y, np.shape(x))

         return y





class high_pass_filter(object):
    ''' This is a low pass filter object that conducts the 1D low pass filtering
        
    '''

    def __init__(self, nsmp, T, cut_freq, transit_freq_length=1.0, axis=0, padding_zeros=False, nl=0, nr=0):
        '''
        Input:
            data - data can be a 1D array or 2D matrix or 3D cubic
            nsmp - number of sample points
            T - the maximum physical time of the signal
            cut_freq - the frequency that uses to cut the signal
            transit_freq_length - the length of frequency that transit the filter value from 0 to 1
            axis - the axis over which to compute the low pass filter, default is -1
            padding_zeros - define if zeros are requried to be padded before and after the signal to avoid the wrapping around affect
            nl - the number of zero-points to be padded at the left side of the signal
            nr - the number of zero-points to be padded at the right side of the signal

        output:
            dataout - the data after the low pass filter
        '''

        self.T_org = T
        self.T_cmp = T / (nsmp-1) * (nsmp+nl+nr-1)
        self.nsmp_org = nsmp
        self.nsmp_cmp = nsmp + nl + nr
        self.cut_freq = cut_freq
        self.transit_freq_length = transit_freq_length
        self.axis = axis
        self.padding_zeros = padding_zeros
        self.shape = [nsmp+nl+nr, nsmp]
        if padding_zeros is True:
            self.padding_zeros_op = padding_zeros_op(self.nsmp_org, nl, nr, axis=axis)

        df = 1/self.T_cmp
        self.df = df

        n_cut = int((cut_freq+transit_freq_length) / df) + 1
        high_pass_filter = np.ones(self.nsmp_cmp)
        low_freq_part = np.zeros(n_cut)
        ind_transit_start = n_cut - int(transit_freq_length / df) 
        ind_transit_end = n_cut - 1
        n_transit = ind_transit_end - ind_transit_start + 1

        x_transit = np.linspace(0.0, np.pi/2.0, n_transit)
        y_transit = np.sin(x_transit)

        low_freq_part[n_cut-n_transit : n_cut] = y_transit
        high_freq_part = np.flip(low_freq_part, axis=-1)

        high_pass_filter[0:n_cut] = low_freq_part
        high_pass_filter[self.nsmp_cmp-n_cut:self.nsmp_cmp] = high_freq_part

        self.high_pass_filter = high_pass_filter 

    def __mul__(self, x):
        
        x_shape = np.shape(x)

        if len(x_shape) == 1:
            if x_shape[0] != self.shape[1]:
                raise ValueError(
                    'The size of input x should be equal to the shape[1] of the high pass filter object')

            else:
               y = self._apply_filter(x) 

        else:
            if x_shape[self.axis] != self.shape[1]:
                raise ValueError(
                    "The length of input x's operating axis should be equal to the shape[1] of the high pass filter object")
            else:
                y = np.apply_along_axis(self._apply_filter, self.axis, x)

        return y

    def __adj_mul__(self, x):
        x_shape = np.shape(x)

        if len(x_shape) == 1:
            if x_shape[0] != self.shape[0]:
                raise ValueError(
                    'The size of input x should be equal to the shape[0] of the high pass filter object for adj_mul')

            else:
               y = self._apply_adj_filter(x)

        else:
            if x_shape[self.axis] != self.shape[0]:
                raise ValueError(
                    "The length of input x's operating axis should be equal to the shape[0] of the high pass filter object for adj_mul")
            else:
                y = np.apply_along_axis(self._apply_adj_filter, self.axis, x)

        return y

    def _apply_filter(self, x):
        if self.padding_zeros is True:
            y = self.padding_zeros_op * x
            y = np.fft.fft(y)
        else:
            y = np.fft.fft(x)

        y = self.high_pass_filter * y
        y = np.fft.ifft(y)
        return y.real

    def _apply_adj_filter(self, x):
        y = np.fft.fft(x)
        y = self.high_pass_filter * y
        y = np.fft.ifft(y)
        y = y.real

        if self.padding_zeros is True:
            y = self.padding_zeros_op.__adj_mul__(y.real)

        return y


class low_pass_filter(object):
    ''' This is a low pass filter object that conducts the 1D low pass filtering
        
    '''

    def __init__(self, nsmp, T, cut_freq, transit_freq_length=1.0, axis=0, padding_zeros=False, nl=0, nr=0):
        '''
        Input:
            data - data can be a 1D array or 2D matrix or 3D cubic
            nsmp - number of sample points
            T - the maximum physical time of the signal
            cut_freq - the frequency that uses to cut the signal
            transit_freq_length - the length of frequency that transit the filter value from 0 to 1
            axis - the axis over which to compute the low pass filter, default is -1
            padding_zeros - define if zeros are requried to be padded before and after the signal to avoid the wrapping around affect
            nl - the number of zero-points to be padded at the left side of the signal
            nr - the number of zero-points to be padded at the right side of the signal

        output:
            dataout - the data after the low pass filter
        '''

        self.T_org = T
        self.T_cmp = T / (nsmp-1) * (nsmp+nl+nr-1)
        self.nsmp_org = nsmp
        self.nsmp_cmp = nsmp + nl + nr
        self.cut_freq = cut_freq
        self.transit_freq_length = transit_freq_length
        self.axis = axis
        self.padding_zeros = padding_zeros
        self.shape = [nsmp+nl+nr, nsmp]
        if padding_zeros is True:
            self.padding_zeros_op = padding_zeros_op(self.nsmp_org, nl, nr, axis=axis)

        df = 1/self.T_cmp
        self.df = df

        n_cut = int((cut_freq+transit_freq_length) / df) + 1
        low_pass_filter = np.zeros(self.nsmp_cmp)
        low_freq_part = np.ones(n_cut)
        ind_transit_start = n_cut - int(transit_freq_length / df)
        ind_transit_end = n_cut - 1
        n_transit = ind_transit_end - ind_transit_start + 1

        x_transit = np.linspace(0.0, np.pi/2.0, n_transit)
        y_transit = np.cos(x_transit)

        low_freq_part[n_cut-n_transit: n_cut] = y_transit
        high_freq_part = np.flip(low_freq_part, axis=-1)

        low_pass_filter[0:n_cut] = low_freq_part
        low_pass_filter[self.nsmp_cmp-n_cut:self.nsmp_cmp] = high_freq_part

        self.low_pass_filter = low_pass_filter

    def __mul__(self, x):

        x_shape = np.shape(x)

        if len(x_shape) == 1:
            if x_shape[0] != self.shape[1]:
                raise ValueError(
                    'The size of input x should be equal to the shape[1] of the low pass filter object')

            else:
               y = self._apply_filter(x)

        else:
            if x_shape[self.axis] != self.shape[1]:
                raise ValueError(
                    "The length of input x's operating axis should be equal to the shape[1] of the low pass filter object")
            else:
                y = np.apply_along_axis(self._apply_filter, self.axis, x)

        return y

    def __adj_mul__(self, x):
        x_shape = np.shape(x)

        if len(x_shape) == 1:
            if x_shape[0] != self.shape[0]:
                raise ValueError(
                    'The size of input x should be equal to the shape[0] of the low pass filter object for adj_mul')

            else:
               y = self._apply_adj_filter(x)

        else:
            if x_shape[self.axis] != self.shape[0]:
                raise ValueError(
                    "The length of input x's operating axis should be equal to the shape[0] of the low pass filter object for adj_mul")
            else:
                y = np.apply_along_axis(self._apply_adj_filter, self.axis, x)

        return y

    def _apply_filter(self, x):
        if self.padding_zeros is True:
            y = self.padding_zeros_op * x
            y = np.fft.fft(y)
        else:
            y = np.fft.fft(x)

        y = self.low_pass_filter * y
        y = np.fft.ifft(y)
        return y.real

    def _apply_adj_filter(self, x):
        y = np.fft.fft(x)
        y = self.low_pass_filter * y
        y = np.fft.ifft(y)
        y = y.real

        if self.padding_zeros is True:
            y = self.padding_zeros_op.__adj_mul__(y.real)

        return y 



class band_pass_filter(object):
    ''' This is a low pass filter object that conducts the 1D low pass filtering
        
    '''

    def __init__(self, nsmp, T, freq_band, transit_freq_length=1.0, axis=0, padding_zeros=False, nl=0, nr=0):
        '''
        Input:
            data - data can be a 1D array or 2D matrix or 3D cubic
            nsmp - number of sample points
            T - the maximum physical time of the signal
            cut_freq - the frequency that uses to cut the signal
            transit_freq_length - the length of frequency that transit the filter value from 0 to 1
            axis - the axis over which to compute the low pass filter, default is -1

        output:
            dataout - the data after the low pass filter
        '''

        self.T_org = T
        self.T_cmp = T / (nsmp-1) * (nsmp+nl+nr-1)
        self.nsmp_org = nsmp
        self.nsmp_cmp = nsmp + nl + nr
        self.freq_band = freq_band
        self.transit_freq_length = transit_freq_length
        self.axis = axis
        self.padding_zeros = padding_zeros
        self.shape = [nsmp+nl+nr, nsmp]
        if padding_zeros is True:
            self.padding_zeros_op = padding_zeros_op(
                self.nsmp_org, nl, nr, axis=axis)

        df = 1/self.T_cmp
        self.df = df

        LPF = low_pass_filter(nsmp, T, freq_band[1], transit_freq_length=transit_freq_length, axis=axis, padding_zeros=padding_zeros, nl=nl, nr=nr)
        HPF = high_pass_filter(nsmp, T, freq_band[0], transit_freq_length=transit_freq_length, axis=axis, padding_zeros=padding_zeros, nl=nl, nr=nr)

        self.band_pass_filter = LPF.low_pass_filter * HPF.high_pass_filter

    def __mul__(self, x):

        x_shape = np.shape(x)

        if len(x_shape) == 1:
            if x_shape[0] != self.shape[1]:
                raise ValueError(
                    'The size of input x should be equal to the shape[1] of the band pass filter object')

            else:
               y = self._apply_filter(x)

        else:
            if x_shape[self.axis] != self.shape[1]:
                raise ValueError(
                    "The length of input x's operating axis should be equal to the shape[1] of the band pass filter object")
            else:
                y = np.apply_along_axis(self._apply_filter, self.axis, x)

        return y

    def __adj_mul__(self, x):
        x_shape = np.shape(x)

        if len(x_shape) == 1:
            if x_shape[0] != self.shape[0]:
                raise ValueError(
                    'The size of input x should be equal to the shape[0] of the band pass filter object for adj_mul')

            else:
               y = self._apply_adj_filter(x)

        else:
            if x_shape[self.axis] != self.shape[0]:
                raise ValueError(
                    "The length of input x's operating axis should be equal to band shape[0] of the low pass filter object for adj_mul")
            else:
                y = np.apply_along_axis(self._apply_adj_filter, self.axis, x)

        return y

    def _apply_filter(self, x):
        if self.padding_zeros is True:
            y = self.padding_zeros_op * x
            y = np.fft.fft(y)
        else:
            y = np.fft.fft(x)
        
        y = self.band_pass_filter * y
        y = np.fft.ifft(y)
        return y.real

    def _apply_adj_filter(self, x):
        y = np.fft.fft(x)
        y = self.band_pass_filter * y
        y = np.fft.ifft(y)
        y = y.real

        if self.padding_zeros is True:
            y = self.padding_zeros_op.__adj_mul__(y.real)

        return y 

def envelope_fun(data, p):
    data_Hilbert = hilbert(data, axis=0).imag
    data_envelope = data**2.0 + data_Hilbert**2.0

    d_data_envelope = p * data_envelope**(p/2.0 - 1.0) * data

    denvelope_ddataH = p * data_envelope**(p/2.0 - 1.0) * data_Hilbert
    d_data_envelope += (-hilbert(denvelope_ddataH, axis=0)).imag
    

    return data_envelope**(p/2.0), d_data_envelope

def correlate_fun(dobs, dpred, mode='fwd'):
    
    # nd = len(dobs)
    a = np.fft.fft(dobs)
    b = np.fft.fft(dpred)
    if mode == 'fwd':
        # output = np.correlate(dobs, dpred, mode='full')
        # output = np.correlate(dpred, dobs, mode='same')
        output = np.fft.ifft(np.conj(a)*b)

        return output.real

    else:
        # output = np.convolve(dpred, dobs, mode='same')
        # output = np.convolve(dobs, dpred, mode='same')
        output = np.fft.ifft(a*b)

        return output.real
        # ndobs = len(dobs)
        # ndpred = len(dpred)
        # output = np.zeros(ndobs)
        
        # for i in range(ndobs):
        #     output[i] = np.dot(dobs, dpred[ndpred-ndobs-i:ndpred-i])


    return output

class padding_zeros_op(object):
    def __init__(self, n_data, nl, nr, axis=0):
        self.n_data = n_data
        self.nl = nl
        self.nr = nr 
        self.shape = [nl+nr+n_data, n_data]
        self.axis = axis

    def __mul__(self, x):
        x_shape = np.shape(x)

        if len(x_shape) == 1:
            if x_shape[0] != self.shape[1]:
                raise ValueError('The size of input x should be equal to shape[1] of the padding_zeros_op operator')

            else:
               y = padding_zeros_fun(x, self.n_data, self.nl, self.nr)

        else:
            if x_shape[self.axis] != self.shape[1]:
                raise ValueError("The length of input x's operating axis should be equal to shape[1] of the padding_zeros_op operator")
            else:
                y = np.apply_along_axis(
                    padding_zeros_fun, self.axis, x, self.n_data, self.nl, self.nr)

        return y

    def __adj_mul__(self, x):
        x_shape = np.shape(x)

        if len(x_shape) == 1:
            if x_shape[0] != self.shape[0]:
                raise ValueError(
                    'The size of input x should be equal to shape[0] of the padding_zeros_op operator')

            else:
               y = un_padding_zeros_fun(x, self.n_data, self.nl, self.nr)

        else:
            if x_shape[self.axis] != self.shape[0]:
                raise ValueError(
                    "The length of input x's operating axis should be equal to shape[1] of the padding_zeros_op operator")
            else:
                y = np.apply_along_axis(
                    un_padding_zeros_fun, self.axis, x, self.n_data, self.nl, self.nr)

        return y



def padding_zeros_fun(data, n_data, nl, nr):
    output = np.zeros(n_data+nl+nr)
    output[nl:nl+n_data] = data

    return output

def un_padding_zeros_fun(data, n_data, nl, nr):
    return data[nl:nl+n_data]

def optimal_transport_fwi(dobs, dpred, dt, transform_mode='linear', c_ratio=5.0, exp_a=1.0, env_p=2.0):

    ## Transform_mode: linear, quadratic, absolute, exponential
    
    # Normalization and transfer data to a distribution
    c = c_ratio * np.abs(np.max(np.abs(dobs)))

        

    if transform_mode == 'linear':          
        # if c < np.abs(np.min(dpred)):
        #     print('c {0}'.format(c))
        #     print('min dpred {0}'.format(np.min(dpred)))

        g = dobs + c
        g = g / (np.sum(g)*dt)
        f_plus_c = dpred + c
        s = (np.sum(f_plus_c)*dt)
        f = f_plus_c / s
    elif transform_mode == 'quadratic':
        g = dobs ** 2.0
        g = g / (np.sum(g)*dt)
        f_quadr = dpred ** 2.0
        s = np.sum(f_quadr)*dt
        f = f_quadr / s
    elif transform_mode == 'absolute':
        g = np.abs(dobs)
        g = g / (np.sum(g)*dt)
        f_abs = np.abs(dpred)
        s = np.sum(f_abs)*dt
        f = f_abs / s
    elif transform_mode == 'exponential':
        g = np.exp(dobs * exp_a)
        g = g / (np.sum(g)*dt)
        f_exp = np.exp(dpred * exp_a)
        s = np.sum(f_exp)*dt
        f = f_exp / s
    elif transform_mode == 'envelope':
        g, df_envo = envelope_fun(dobs, env_p)
        g = g / (np.sum(g)*dt)
        f_env, df_env = envelope_fun(dpred, env_p)
        s = np.sum(f_env)*dt
        f = f_env / s

    if np.min(f) < 0:
        resid = 1e10
        adj_src = np.zeros(dpred.shape)
        # print('small c used')
        return resid, adj_src, np.linalg.norm(resid)**2.0


    f[np.where(np.abs(f)<1e-20)] = 0.0
    g[np.where(np.abs(g)<1e-20)] = 0.0
    ndata = len(f)
    t = np.array(range(0,ndata)) * dt
    
    if transform_mode == 'linear': 
        if c < np.abs(np.min(dpred)):
            print('min f {0}'.format(np.min(f)))
            print('min g {0}'.format(np.min(g)))

    F = np.zeros(ndata)
    G = np.zeros(ndata)
    g_IGoF = np.zeros(ndata)
    IGoF = np.zeros(ndata)
    IGoF_ind = np.zeros(ndata)

    # Compute F(t) and G(t)
    int_f = 0.0
    int_g = 0.0
    for i in range(0, ndata):
        int_f += f[i]
        int_g += g[i]
        F[i] = int_f
        G[i] = int_g

    # Compute G^{-1} o F(t)
    # IGoF[ndata-1] = (ndata-1)*dt
    # IGoF_ind[ndata-1] = ndata-1

    IGoF = np.interp(F, G, t)
    g_IGoF = np.interp(IGoF, t, g)

    # IGoF_ind = IGoF_ind.astype(int)
    # # F[ndata-1] = G[ndata-1]
    # F[np.where(F>G[ndata-1])] = G[ndata-1]-10**(-10)
    # for i in range(0, ndata):

    #     IGoF_ind[i] = int(np.searchsorted(G, F[i]))
    #     if IGoF_ind[i] == ndata:
    #         # print('F: ',F)
    #         # print('G: ',G[ndata-1])
    #         # print('Dpred: ', dpred)
    #         IGoF_ind[i] = ndata-1

    #     if IGoF_ind[i] == 0:
    #         IGoF[i] = IGoF_ind[i] * dt
    #         g_IGoF[i] = g[0]
    #     else:
    #         beta = (G[IGoF_ind[i]] - F[i]) / (G[IGoF_ind[i]] - G[IGoF_ind[i]-1])
    #         IGoF[i] = (IGoF_ind[i] - beta) * dt
    #         g_IGoF[i] = g[IGoF_ind[i]] - beta * (g[IGoF_ind[i]] - g[IGoF_ind[i]-1])


    # idx = np.where(IGoF_ind>ndata-1)
    # IGoF_ind[idx] = ndata-1
    # if c < np.abs(np.min(dpred)):
    #     print('min IGoF_ind {0}'.format(np.min(IGoF_ind)))
    #     print('max IGoF_ind {0}'.format(np.max(IGoF_ind)))

    # Compute residual
    t_minus_IGoF = t - IGoF
    resid = np.sqrt(f) * t_minus_IGoF
#    print('min resid {0}'.format(np.min(resid)))
#    print('max resid {0}'.format(np.max(resid)))

    # Compute adjoint source

    adj_src1 = t_minus_IGoF * t_minus_IGoF
    # adj_src2 = (-2*f*dt / g[IGoF_ind]) * t_minus_IGoF
    f_divid_g = np.zeros(ndata)
    idx_gnot0 = np.where(g_IGoF > 0)
    f_divid_g[idx_gnot0] = -2*f[idx_gnot0]*dt / g_IGoF[idx_gnot0]

    adj_src2 = f_divid_g * t_minus_IGoF

    for i in range(ndata-2, -1, -1):
        adj_src2[i] += adj_src2[i+1]

    adj_src = adj_src1+adj_src2
    

    if transform_mode == 'linear':
        adj_src = adj_src / s - (dt/(s**2.0)*np.dot(f_plus_c, adj_src))*np.ones(ndata)
    elif transform_mode == 'quadratic':
        adj_src = (adj_src / s - (dt/(s**2.0)*np.dot(f_quadr, adj_src))*np.ones(ndata))*2.0*dpred
        # adj_src = adj_src / s - (dt/(s**2.0)*np.dot(f_quadr, adj_src))*2.0*dpred
    elif transform_mode == 'absolute':
        adj_src = (adj_src / s - (dt/(s**2.0)*np.dot(f_abs, adj_src))*np.ones(ndata))*np.sign(dpred)
    elif transform_mode == 'exponential':
        adj_src = (adj_src / s - (dt/(s**2.0)*np.dot(f_exp, adj_src))*np.ones(ndata))*(np.exp(dpred*exp_a) * exp_a)
    elif transform_mode == 'envelope':
        adj_src = (adj_src / s - (dt/(s**2.0)*np.dot(f_env, adj_src)) * np.ones(ndata))*(df_env)

    return resid, adj_src, np.linalg.norm(resid)**2.0


if __name__ == '__main__':
    
    import numpy as np

    nsmp = [200,100]
    A = np.random.normal(0,1,nsmp)
    S = opSmooth2D(nsmp, [100,100], window_len=[5,5])
    B = S * np.reshape(A,(S.nsmp,1))
    B = np.reshape(B, nsmp)

    plt.figure()
    plt.imshow(A)
    plt.show()
    plt.figure()
    plt.imshow(B)
    plt.show()

    nsmp = 501
    nshift = 101
    dt = 0.01
    T = (nsmp-1)*dt
    xt = np.linspace(0, T, nsmp)
    freqt = np.linspace(0, 100, nsmp)
    a = 2.0
    f0 = signal.ricker(nsmp, a)
    F0 = np.zeros((nsmp,2))
    F0[:,0] = f0
    F0[:,1] = f0
    S = opSmooth1D(nsmp,window_len=50)
    F1 = S * F0
    plt.figure()
    plt.plot(F1[:,0])
    plt.show()
    plt.figure()
    plt.plot(F1[:,0])
    plt.show()


    nsmp = 501
    nshift = 101
    dt = 0.01
    T = (nsmp-1)*dt
    xt = np.linspace(0, T, nsmp)
    freqt = np.linspace(0, 100, nsmp)
    a = 2.0
    f0 = signal.ricker(nsmp, a)
    f1 = np.roll(f0,5)
    resid, adj_src, obj1 = optimal_transport_fwi(f0, f1, 1, transform_mode='linear')
    resid, adj_src, obj2 = optimal_transport_fwi(f1, f0, 1, transform_mode='linear')
    print('obj1: ', obj1)
    print('obj2: ', obj2)
    f1 = np.roll(f0, -5)
    resid, adj_src, obj1 = optimal_transport_fwi(f0, f1, 1, transform_mode='linear')
    resid, adj_src, obj2 = optimal_transport_fwi(f1, f0, 1, transform_mode='linear')
    print('obj1: ', obj1)
    print('obj2: ', obj2)

    resid, adj_src, obj1 = optimal_transport_fwi(f0, f1, 1, transform_mode='absolute')
    resid, adj_src, obj2 = optimal_transport_fwi(f1, f0, 1, transform_mode='quadratic')
    print('obj_abs: ', obj1)
    print('obj_qua: ', obj2)

    obj_quadr = np.zeros(nshift)
    obj_linear = np.zeros(nshift)
    obj_abs = np.zeros(nshift)
    obj_exp = np.zeros(nshift)
    for i in range(nshift):
        k = i - (nshift-1) // 2
        f1 = np.roll(f0,k)
        resid, adj_src, obj_linear[i] = optimal_transport_fwi(f0, f1, 1, transform_mode='linear')
        resid, adj_src, obj_quadr[i] = optimal_transport_fwi(f0, f1, 1, transform_mode='quadratic')
        resid, adj_src, obj_abs[i] = optimal_transport_fwi(f0, f1, 1, transform_mode='absolute')
        resid, adj_src, obj_exp[i] = optimal_transport_fwi(f0, f1, 1, transform_mode='exponential')
    
    plt.figure()
    plt.plot(obj_linear/np.max(obj_linear), label='linear')
    plt.plot(obj_quadr/np.max(obj_quadr), label='quadratic')
    plt.plot(obj_abs/np.max(obj_abs), label='absolute')
    plt.plot(obj_exp/np.max(obj_exp), label='exponential')
    plt.legend()
    plt.show()


    nsmp = 9001
    ntest = 100
    for i in range(ntest):
        a = np.random.normal(0,1,nsmp)
        b = np.random.normal(0,1,nsmp)
        resid, adj_src, obj1 = optimal_transport_fwi(a, b, 1, transform_mode='linear')
        resid, adj_src, obj2 = optimal_transport_fwi(b, a, 1, transform_mode='linear')
        # print(np.abs(obj1-obj2)/obj1)
        print('obj1: ', obj1)
        print('obj2: ', obj2)

    quit

    ## Adjoint test for correlation 

    n_data = 8000
    a_obs = np.random.normal(0,1,n_data)
    b = np.random.normal(0,1,n_data)
    c = np.random.normal(0, 1, n_data)

    print(np.dot(c, correlate_fun(a_obs,b)))
    print(np.dot(correlate_fun(a_obs,c,mode='adj'), b))

    # ## Test padding zeros op

    # n_data = 11
    # nr = 3
    # nl = 5
    # a = np.random.normal(0,1,(n_data,2))
    # J = padding_zeros_op(n_data, nl, nr)
    # b = J * a
    # c = J.__adj_mul__(b)

    # print(b)
    # print(c)

    # ## Test optimal transport 
    
    # a = np.array([0,0,-1,0,1,0,-1,0,0])
    # b = np.array([0,-1,0,1,0,-1,0,0,0])
    # dt = 0.2
    # resid, adj_src, r = optimal_transport_fwi(a, b, dt)
    
    # nd = 11
    # a = np.random.normal(0,1,nd)
    # b = np.random.normal(0,1,nd*2-1)
    # c = np.random.normal(0,1,nd)
    
    # d = np.dot(b, correlate_fun(a,c))
    # e = np.dot(c, correlate_fun(a,b,mode='adj'))
    # print('d = {0}'.format(d))
    # print('e = {0}'.format(e))
    
    # o = [0.0, 1.0, 2.0]
    # d = [1.0, 2.0, 3.0]
    # n = [4, 5, 6]
    # output = odn2grid(o, d, n)
    # print(output[0])
    # print(output[1])
    # print(output[2])

    # o = [0.0, 1.0, 2.0, 3.0, 4.0]
    # d = [1.0, 2.0, 3.0, 4.0, 5.0]
    # n = [4, 1, 5, 1, 6]
    # data_time, data_xrec, data_zrec, data_xsrc, data_zsrc = odn2grid_data_2D_time(o, d, n)

    # print(data_time)
    # print(data_zrec)
    # print(data_xrec)
    # print(data_zsrc)
    # print(data_xsrc)

    nsmp = 501
    dt = 0.01
    T = (nsmp-1)*dt 
    xt = np.linspace(0, T, nsmp)
    freqt = np.linspace(0, 100, nsmp)
    a = 6.0
    f0 = signal.ricker(nsmp, a)
    f1 = np.zeros(f0.shape)
    f1[0:400] = f0[100:500]
    f0 = f1
    cut_freq = 0.2
    cut_freqh = 0.5
    freq_band = [0.2, 1.0]
    nl = 200
    nr = 100
    xt2 = np.linspace(-nl*dt, T+nr*dt, nsmp+nl+nr)


    # LF = low_pass_filter(nsmp, T, cut_freq, padding_zeros=True, nl=nl, nr=nr)
    # f1 = LF * f0

    # plt.plot(xt,f0)
    # plt.plot(xt,f1)
    # plt.show()
    # plt.figure()
    # plt.plot(freqt, np.abs(np.fft.fft(f0)))
    # plt.plot(freqt, np.abs(np.fft.fft(f1)))
    # plt.show()

    # HF = high_pass_filter(nsmp, T, cut_freqh, padding_zeros=True, nl=nl, nr=nr)
    # f1 = HF * f0

    # plt.plot(xt, f0)
    # plt.plot(xt, f1)
    # plt.show()
    # plt.figure()
    # plt.plot(freqt, np.abs(np.fft.fft(f0)))
    # plt.plot(freqt, np.abs(np.fft.fft(f1)))
    # plt.show()

    BF = band_pass_filter(nsmp, T, freq_band, padding_zeros=True, nl=nl, nr=nr)

    a = np.random.normal(0,1, BF.shape[0])
    b = np.random.normal(0, 1, BF.shape[1])

    print(np.inner(a, BF*b))
    print(np.inner(BF.__adj_mul__(a),b))

    f1 = BF * f0
    f2 = BF.__adj_mul__(f1)

    plt.plot(xt, f0)
    plt.figure()
    plt.plot(xt2, f1)
    plt.figure()
    plt.plot(xt, f2)
    plt.show()
    plt.figure()
    plt.plot(freqt, np.abs(np.fft.fft(f0)))
    plt.plot(freqt, np.abs(np.fft.fft(f1)))
    plt.show()


    f0=f0.reshape((-1,1))
    F0 = np.concatenate((f0,f0),axis=1)
    F1 = LF * F0

    a = 1

        



