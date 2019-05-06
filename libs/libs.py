import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


# Calculate LSS(overall)
def LSS_overall_cal(y):
    overall_prob = len(np.where(y == 1)[0]) / len(y)
    LSS_overall = np.sum(np.where(y == 1, np.log(overall_prob + 1e-7), np.log(1 - overall_prob + 1e-7)))
    return LSS_overall

# Calculate LSS(fit)
def LSS_fit_cal(classifier, x, y):
    classifier.fit(x, y)
    y_hat = classifier.predict_proba(x)
    LSS_fit = np.sum(np.where(y == 1, np.log(y_hat[:, 1] + 1e-7), np.log(y_hat[:, 0] + 1e-7)))
    return LSS_fit

# Calculate TSS
def TSS_cal(y):
    y_mean = np.mean(y)
    TSS = np.sum(np.square(y - y_mean))
    return TSS

# Calculate RSS
def RSS_cal(regression, x, y):
    regression.fit(x, y)
    y_hat = regression.predict(x)
    RSS = np.sum(np.square(y - y_hat))
    return RSS

# Calculate F-statistic
def F_test(x, y, datatype="categorical", threshold=5):
    assert datatype == "categorical" or datatype == "numerical"

    n = x.shape[0]
    p = x.shape[1]
    if datatype == "categorical":
        LSS_overall = LSS_overall_cal(y)
        print("LSS_overall: {:.2f}".format(LSS_overall))
        LSS_fit = LSS_fit_cal(LogisticRegression(solver="lbfgs", C=1e10, max_iter=1000), x, y)
        print("LSS_fit: {:.2f}".format(LSS_fit))
        F_statistic = ((LSS_overall - LSS_fit) / p) / (LSS_fit / (n - p - 1))
    else:
        TSS = TSS_cal(y)
        print("TSS: {:.2f}".format(TSS))
        RSS = RSS_cal(LinearRegression(), x, y)
        print("RSS: {:.2f}".format(RSS))
        F_statistic = ((TSS - RSS) / p) / (RSS / (n - p - 1))

    if F_statistic >= threshold:
        print("F-test score is {:.2f}, that is more than {}, therefore, there is relationship between predictors and response.".format(F_statistic, threshold))
    else:
        print("F-test score is {:.2f}, that is less than {}, therefore, there is no relationship between predictors and response.".format(F_statistic, threshold))

# Calculate R^2 for
def R_squared(x, y, datatype="categorical"):
    assert datatype == "categorical" or datatype == "numerical"

    if datatype == "categorical":
        LSS_fit = LSS_fit_cal(LogisticRegression(solver="lbfgs", C=1e10, max_iter=1000), x, y)
        LSS_overall = LSS_overall_cal(y)
        R_squared = 1 - (LSS_fit / LSS_overall)
    else:
        TSS = TSS_cal(y)
        RSS = RSS_cal(LinearRegression(), x, y)
        R_squared = 1 - (RSS / TSS)

    return R_squared
