import pandas as pd
import numpy as np
from pykalman import KalmanFilter

kf = KalmanFilter(transition_covariance=0.18, observation_covariance=1)


def load_cgm_data(filename, col_name):
    df = pd.read_csv(filename, encoding='utf-8')
    return load_cgm_data(df, col_name)


def load_cgm_data(df, col_name):
    indices = np.array(df['index'].to_list())
    col_datas = np.array(df[col_name].to_list())

    return np.append(np.zeros(indices[0]), col_datas)


def parse_data(data, data_range, in_num, set_ratio):
    data = min_max_normalize(data[data_range[0]:data_range[1] + 1])

    x, y = np.array([]), np.array([])

    for i in range(len(data) - in_num):
        x = np.append(x, data[i:i + in_num])
        y = np.append(y, data[i + in_num])

    x = np.array(x).reshape(int(len(x) / in_num), in_num)

    return x[:int(len(x) * set_ratio)], x[int(len(x) * set_ratio):], \
           y[:int(len(y) * set_ratio)], y[int(len(y) * set_ratio):]


def min_max_normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def kalman_filter(measurements):
    (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)

    return filtered_state_means


