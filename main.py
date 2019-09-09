import pandas as pd
from data_process import *
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# setting
df_bgm = pd.read_csv('149-A7020501/bgm_record_20190815104645.csv', encoding='utf-8')
df_cgm = pd.read_csv('149-A7020501/T2310EED0149_20190815104645.csv', encoding='utf-8')

# df_cgm = pd.read_csv('cgm_data.csv', encoding='utf-8')
cgm_data_range = [5400, 10200]
testing_range = [5415, 6000]
in_num = 15
a_b = [13.939655, -10.574911]


# cgm_data
cgm_data = load_cgm_data('electricCurrent1', df_cgm)[cgm_data_range[0]:cgm_data_range[1]+1]
cgm_data_min = np.min(cgm_data)
cgm_data_max = np.max(cgm_data)
cgm_data = min_max_normalize(cgm_data, cgm_data_min, cgm_data_max)
train_x, test_x, train_y, test_y = parse_data(
    cgm_data,
    in_num,
    [testing_range[0] - cgm_data_range[0], testing_range[1] - cgm_data_range[0]])



# filter_data
filter_data = kalman_filter(cgm_data)
train_f_x, test_f_x, train_f_y, test_f_y = parse_data(
    filter_data,
    in_num,
    [testing_range[0] - cgm_data_range[0], testing_range[1] - cgm_data_range[0]])

# bgm_data
bgm_mgdl = df_bgm['mgdl'].to_list()
bgm_index = df_bgm['targetIndex'].to_list()
bgm_mgdl_inrange = []
bgm_index_inrange = []

for i in range(len(bgm_index)):
    if testing_range[1] >= bgm_index[i] >= testing_range[0]:
        bgm_mgdl_inrange.append(bgm_mgdl[i])
        bgm_index_inrange.append(bgm_index[i])


svr_cgm = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_cgm.fit(train_x, train_y)


svr_filter = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_filter.fit(train_f_x, train_f_y)

# plt.figure(num=None, figsize=(24, 6), dpi=30, facecolor='w', edgecolor='k')


plt.plot(range(testing_range[0], testing_range[1]+1),
         currents_to_glucose(
             normalize_inverse(test_y, cgm_data_min, cgm_data_max), a_b[0], a_b[1]),
         label='source')
plt.plot(range(testing_range[0], testing_range[1]+1),
         currents_to_glucose(
             normalize_inverse(test_f_y, cgm_data_min, cgm_data_max), a_b[0], a_b[1]),
         label='filter source')
plt.plot(range(testing_range[0], testing_range[1]+1),
         currents_to_glucose(
             normalize_inverse(svr_cgm.predict(test_x), cgm_data_min, cgm_data_max), a_b[0], a_b[1]),
         label='predict')
plt.plot(range(testing_range[0], testing_range[1]+1),
         currents_to_glucose(
             normalize_inverse(svr_filter.predict(test_f_x), cgm_data_min, cgm_data_max), a_b[0], a_b[1]),
         label='filter predict')


# plt.plot(range(cgm_data_range[0], cgm_data_range[1]+1),
#          currents_to_glucose(
#              normalize_inverse(cgm_data[cgm_data_range[0]: cgm_data_range[1]+1], cgm_data_min, cgm_data_max),
#              13.939655,
#              -10.574911
#          ), label='source')
# plt.plot(range(cgm_data_range[0], cgm_data_range[1]+1),
#          currents_to_glucose(
#              normalize_inverse(filter_data[100:], cgm_data_min, cgm_data_max),
#              13.939655,
#              -10.574911
#          ), label='filter')
plt.plot(bgm_index_inrange, bgm_mgdl_inrange, 'o', label='bgm')
plt.legend()
# plt.xticks(range(cgm_data_range[0], cgm_data_range[1]+1, 100))
plt.show()

