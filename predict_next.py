import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def load_cgm_data(in_num, train_set_ratio):
    data_range = [5400, 9000]

    df = pd.read_csv('cgm_data.csv', encoding='utf-8')

    indices = np.array(df['index'].to_list())
    currents = np.array(df['electricCurrent1'].to_list())
    # currents = np.array(df['換算濃度'].to_list())
    trans = np.array(df['換算濃度'].to_list())

    offset_currents = np.append(np.zeros(indices[0]), currents)
    offset_trans = np.append(np.zeros(indices[0]), trans)

    currents = offset_currents[data_range[0]:data_range[1]+1]
    currents = (currents - np.min(currents))/(np.max(currents)-np.min(currents))

    trans = offset_trans[data_range[0]:data_range[1] + 1]
    trans = (trans - np.min(trans)) / (np.max(trans) - np.min(trans))

    x, y = np.array([]), np.array([])

    for i in range(len(currents) - in_num):
        x = np.append(x, currents[i:i+in_num])
        y = np.append(y, currents[i+in_num])

    x = np.array(x).reshape(int(len(x)/in_num), in_num)

    return x[:int(len(x)*train_set_ratio)], x[int(len(x)*train_set_ratio):],\
           y[:int(len(y)*train_set_ratio)], y[int(len(y)*train_set_ratio):],\
           trans[int(len(y)*train_set_ratio)+10:]


def main():
    train_x, test_x, train_y, test_y, trans = load_cgm_data(10, 0.95)
    print(test_y)
    print(trans)

    # #############################################################################
    # Fit regression model
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr_lin = SVR(kernel='linear', C=100, gamma='auto')
    svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
                   coef0=1)

    # #############################################################################
    # Look at the results
    lw = 2

    # svrs = [svr_rbf, svr_lin, svr_poly]
    # kernel_label = ['RBF', 'Linear', 'Polynomial']
    # model_color = ['m', 'c', 'g']
    #
    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)

    svrs = [svr_rbf]
    kernel_label = ['RBF']
    model_color = ['m']

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 10), sharey=True)
    for ix, svr in enumerate(svrs):
        svr.fit(train_x, train_y)
        axes.plot(range(len(test_x)), svr.predict(test_x), color=model_color[ix], lw=lw,
                      label='{} model'.format(kernel_label[ix]))

        axes.plot(range(len(test_y)), test_y, color='c', lw=lw,
                  label='data source')

        axes.plot(range(len(test_y)), trans, color='g', lw=lw,
                  label='transform')

        # axes.scatter(X[svr.support_], y[svr.support_], facecolor="none",
        #                  edgecolor=model_color[ix], s=50,
        #                  label='{} support vectors'.format(kernel_label[ix]))
        #
        # axes.scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
        #                  y[np.setdiff1d(np.arange(len(X)), svr.support_)],
        #                  facecolor="none", edgecolor="k", s=50,
        #                  label='other training data')

        axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                        ncol=1, fancybox=True, shadow=True)
    print(mean_squared_error(test_y, svr.predict(test_x)))
    fig.text(0.5, 0.04, 'index', ha='center', va='center')
    fig.text(0.06, 0.5, 'concentration', ha='center', va='center', rotation='vertical')
    fig.suptitle("Support Vector Regression", fontsize=14)
    plt.show()


if __name__ == '__main__':
    main()
