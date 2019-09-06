import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


def load_cgm_data():
    df = pd.read_csv('cgm_data.csv', encoding='utf-8')
    df_range = pd.read_csv('ranges.csv', encoding='utf-8')

    indices = np.array(df['index'].to_list())
    currents = np.array(df['換算濃度'].to_list())

    starts = np.array(df_range['start'].to_list())
    ends = np.array(df_range['end'].to_list())

    offset_currents = np.append(np.zeros(indices[0]), currents)

    x, y = [],[]

    for i in range(len(starts)):
        if ends[i] - starts[i] >= 0:
            y = y + list(offset_currents[starts[i]: ends[i]+1])
            x = x + list(range(ends[i] - starts[i] + 1))

    return np.array(x).reshape(len(x), 1), np.array(y)


def main():
    X, y = load_cgm_data()
    test_X = np.arange(np.max(X))
    test_X = test_X.reshape(len(test_X), 1)

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
        svr.fit(X, y)
        axes.plot(test_X, svr.predict(test_X), color=model_color[ix], lw=lw,
                      label='{} model'.format(kernel_label[ix]))

        axes.scatter(X[svr.support_], y[svr.support_], facecolor="none",
                         edgecolor=model_color[ix], s=50,
                         label='{} support vectors'.format(kernel_label[ix]))

        axes.scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                         facecolor="none", edgecolor="k", s=50,
                         label='other training data')

        axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                        ncol=1, fancybox=True, shadow=True)

    fig.text(0.5, 0.04, 'index', ha='center', va='center')
    fig.text(0.06, 0.5, 'concentration', ha='center', va='center', rotation='vertical')
    fig.suptitle("Support Vector Regression", fontsize=14)
    plt.show()


if __name__ == '__main__':
    main()
