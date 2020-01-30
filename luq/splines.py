# Copyright 2019 Steven Mattis and Troy Butler
from scipy import optimize
import numpy as np

# Splines and other methods for cleaning data.


def linear_C0_spline(times, data, num_knots, clean_times):
    """
    Clean a time series over window with C0 linear splines.
    :param times: time window
    :param data: time series data
    :param num_knots: number of knots to use
    :param clean_times: number of clean values wanted
    :return:
    """

    def wrapper_fit_func(x, N, *args):
        Qs = list(args[0][0:N])
        knots = list(args[0][N:])
        return piecewise_linear(x, knots, Qs)

    def wrapper_fit_func_Qs(x, knots, *args):
        Qs = list(args[0])
        return piecewise_linear(x, knots, Qs)

    def piecewise_linear(x, knots, Qs):
        knots = np.insert(knots, 0, 0)
        knots = np.append(knots, 1)
        return np.interp(x, knots, Qs)
    
    def piecewise_linear_clean(x, knots, Qs):
        knots = np.insert(knots, 0, times[0])
        knots = np.append(knots, times[-1])
        return np.interp(x, knots, Qs)
    
    knots_init = np.linspace(0, 1, num_knots)[1:-1]
    
    param_bounds = np.zeros((2, 2*num_knots-2))
    for i in range(2*num_knots-2):
        if i < num_knots:
            param_bounds[:, i] = [-np.inf, np.inf]
        else:
            param_bounds[:, i] = [0, 1]
     
    # find piecewise linear splines for predictions
    try:
        q_pl, _ = optimize.curve_fit(lambda x, *params_0: wrapper_fit_func(x, num_knots, params_0),
                                     (times-times[0])/(times[-1]-times[0]),
                                     data,
                                     p0=np.hstack([np.zeros(num_knots), knots_init]),
                                     bounds=param_bounds)
    except RuntimeError:
        # Use uniform knots
        print('Optimization of knot locations failed. Using uniform knots.')
        knots = np.linspace(0, 1, num_knots)[1:-1]
        q_pl, _ = optimize.curve_fit(lambda x, *params_0: wrapper_fit_func_Qs(x, knots, params_0),
                                     (times-times[0])/(times[-1]-times[0]),
                                     data,
                                     p0=np.zeros(num_knots))
        q_pl = np.hstack([q_pl, knots])

    q_pl[num_knots:] *= (times[-1]-times[0])
    q_pl[num_knots:] += times[0]

    # fix if knots get out of order
    inds_sort = np.argsort(q_pl[num_knots:])
    q_pl[1:num_knots-1] = q_pl[inds_sort+1]  # Qs
    q_pl[num_knots:] = q_pl[inds_sort + num_knots]  # knots

    # calculate clean data
    clean_data = piecewise_linear_clean(clean_times,
                                        q_pl[num_knots:],
                                        q_pl[0:num_knots])

    # calculate mean absolute error between spline and original data
    clean_data_at_original = piecewise_linear_clean(times,
                                                    q_pl[num_knots:],
                                                    q_pl[0:num_knots])
    
    error = np.average(np.abs(clean_data_at_original - data))
    error = error / np.average(np.abs(data))
    return clean_data, error, q_pl




