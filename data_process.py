import pandas as pd
import numpy as np
from pykalman import KalmanFilter

kf = KalmanFilter(transition_covariance=0.18, observation_covariance=1)


def load_cgm_data(col_name, df=None, filename=None):
    if df is None:
        df = pd.read_csv(filename, encoding='utf-8')

    indices = np.array(df['index'].to_list())
    col_datas = np.array(df[col_name].to_list())

    return np.append(np.zeros(indices[0]), col_datas)


def parse_data(data, in_num, testing_range=None):
    x, y = np.array([]), np.array([])

    for i in range(len(data) - in_num):
        x = np.append(x, data[i:i + in_num])
        y = np.append(y, data[i + in_num])

    x = np.array(x).reshape(int(len(x) / in_num), in_num)

    return np.append(x[:testing_range[0]-in_num], x[testing_range[1]-in_num+1:], axis=0), \
           x[testing_range[0]-in_num:testing_range[1]-in_num+1], \
           np.append(y[:testing_range[0]-in_num], y[testing_range[1]-in_num+1:], axis=0), \
           y[testing_range[0]-in_num:testing_range[1]-in_num+1]


def min_max_normalize(data):
    return min_max_normalize(data, np.min(data), np.max(data))


def min_max_normalize(data, data_min, data_max):
    return (data - data_min) / (data_max - data_min)


def normalize_inverse(data, data_min, data_max):
    return data * (data_max - data_min) + data_min


def currents_to_glucose(currents, a, b):
    return currents * a + b


def kalman_filter(measurements):
    (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)

    return filtered_state_means


