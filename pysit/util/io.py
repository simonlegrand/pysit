import numpy as np

import scipy.io as sio

import obspy.io.segy.core as segy

__all__ = ['read_model', 'read_data', 'write_data',
           'write_gathered_parallel_data_time',
           'read_data_2D_parallel_env']


def read_model(fname):
    """ Reads a model in segy format and returns it as an array."""

    # data = segy.readSEGY(fname)
    data = segy._read_segy(fname)

    return np.array([tr.data for tr in data.traces])


def read_data(fname):
    """"
        Reads a data in mat format and returns it as a matrix

        The input of the file should be a dictionary with {'o'=
                                                           'd'=
                                                           'n'=
                                                           'data'=}
        'o' - the origin of each dimension
        'd' - the delta of each dimension
        'n' - the number of points in each dimension
        'data' - the data itself 
    """

    b_dict = sio.loadmat(fname)
    o = b_dict['o'][0]
    d = b_dict['d'][0]
    n = b_dict['n'][0]
    data = b_dict['data']

    return data, o, d, n


def read_data_2D_parallel_env(fname, pwrap):
    """"
        Reads a data in mat format and returns it as a matrix

        The input of the file should be a dictionary with {'o'=
                                                           'd'=
                                                           'n'=
                                                           'data'=}
        'o' - the origin of each dimension
        'd' - the delta of each dimension
        'n' - the number of points in each dimension
        'data' - the data itself 
    """

    if pwrap.comm.rank == 0:
        [data, odata, ddata, ndata] = read_data(fname)
        broad_info = {'o': odata, 'd': ddata, 'n': ndata}
        data1 = np.squeeze(data)
        for i in range(ndata[3]):
            print(i, 'th shot is ', np.linalg.norm(data1[:, :, i]))
    else:
        data = None
        broad_info = None

    broad_info = pwrap.comm.bcast(broad_info, root=0)
    if pwrap.comm.rank is not 0:
        odata = broad_info['o']
        ddata = broad_info['d']
        ndata = broad_info['n']

    return data, odata, ddata, ndata


def write_data(fname, data, o, d, n, label='None'):

        # Wirte a data file in mat format

        # Input:
        # fname - the name of the file
        # data - the data itself
        # o - the origin of each dimension
        # d - the delta of each dimension
        # n - the number of points in each dimension
        # label - label of each dimension

    a_dict = {'o': o, 'd': d, 'n': n, 'data': data, 'label': label}
    sio.savemat(fname, a_dict)


def write_gathered_parallel_data_time(fname, shots_gathered):
    # The order of final output is [time, xrec, zrec, xsrc, zsrc]

    n_15 = shots_gathered[0][0].receivers.data.shape

    k = 0
    for i in range(len(shots_gathered)):
        for j in range(len(shots_gathered[i])):
            k += 1

    n = (n_15[0], n_15[1], 1, k, 1)
    o = (0,
         shots_gathered[0][0].receivers.receiver_list[0].position[0],
         shots_gathered[0][0].receivers.receiver_list[0].position[1],
         shots_gathered[0][0].sources.position[0],
         shots_gathered[0][0].sources.position[1]
         )

    d0 = shots_gathered[0][0].receivers.ts[1] - \
        shots_gathered[0][0].receivers.ts[0]
    d1 = shots_gathered[0][0].receivers.receiver_list[1].position[0] - \
        shots_gathered[0][0].receivers.receiver_list[0].position[0]
    d2 = 1
    if len(shots_gathered[0]) > 1:
        d3 = shots_gathered[0][1].sources.position[0] - \
            shots_gathered[0][0].sources.position[0]
    else:
        d3 = shots_gathered[1][0].sources.position[0] - \
            shots_gathered[0][0].sources.position[0]
    d4 = 1

    d = (d0, d1, d2, d3, d4)

    data = np.zeros(n)
    k = 0
    n_sub = (n[0], n[1], n[2], n[4])

    for i in range(len(shots_gathered)):
        for j in range(len(shots_gathered[i])):
            data[:, :, :, k, :] = shots_gathered[i][j].receivers.data.reshape(
                n_sub)
            k += 1

    label = ['time', 'x_receiver', 'z_receiver', 'x_source', 'z_source']

    write_data(fname, data, o, d, n, label=label)
