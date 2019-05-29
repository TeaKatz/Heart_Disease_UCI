import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures


def TSS(y):
    y = np.array(y)

    return np.sum(np.square(y - np.mean(y)))


def RSS(y, y_hat):
    y = np.array(y)
    y_hat = np.array(y_hat)

    return np.sum(np.square(y - y_hat))


def R_square(y, y_hat):
    y = np.array(y)
    y_hat = np.array(y_hat)

    return 1 - RSS(y, y_hat) / TSS(y)


def evaluation(model, x, y, scoring="accuracy", plot_coef=True):
    features_list = x.columns
    x = np.array(x)
    y = np.array(y)

    model.fit(x, y)
    acc = model.score(x, y)
    print("Train set accuracy: {:.2f}%".format(np.mean(acc) * 100))

    score = cross_val_score(model, x, y, cv=10, scoring=scoring)
    print("Validation set {}: {:.2f}%".format(scoring, np.mean(score) * 100))

    y_hat = model.predict(x)
    print("R_squared: {:.2f}%".format(R_square(y, y_hat) * 100))

    coefficients = pd.Series(model.coef_.ravel(), index=features_list)
    if "intercept" not in features_list:
        intercept = pd.Series(model.intercept_, index=["intercept"])
        coefficients = pd.concat([intercept, coefficients])
    coefficients = coefficients.sort_values(ascending=False)

    if plot_coef:
        coefficients.plot(kind="barh", figsize=(9, 9))
        plt.title("Coefficients")
        plt.show()

    return coefficients


def compare_categorical(x, y):
    assert len(x) == len(y), "length of arrays are not equal: {} != {}".format(len(x), len(y))

    x = np.array(x)
    y = np.array(y)

    n = np.sum(np.equal(x, 1))
    match = np.sum(np.where(x == 1, np.equal(x, y), 0))
    invert_match = np.sum(np.where(x == 1, np.not_equal(x, y), 0))

    print("match: {}/{} ({:.2f}%)".format(match, n, match / n * 100))
    print("invert_match: {}/{} ({:.2f}%)".format(invert_match, n, invert_match / n * 100))


def forward_selection(x, y, scoring="accuracy", keep_first=True):
    features_list = list(x.columns.values)
    iteration = len(features_list)

    optimal_features_sets = []
    max_score = 0
    max_score_features = None

    for _ in range(iteration):
        add_feature = None
        max_R_squared = -1.0
        for candidate_feature in features_list:
            # Prevent duplicate features
            if candidate_feature in optimal_features_sets:
                continue

            # Define set of features to feed to regression model
            selected_features = optimal_features_sets + [candidate_feature]

            # Regression y onto selected features
            logis_reg = LogisticRegression(solver="lbfgs", C=1e10, max_iter=100000)
            logis_reg.fit(x[selected_features], y)
            y_hat = logis_reg.predict(x[selected_features])

            # Calculate R square
            R_squared = R_square(y, y_hat)

            # Get maximum R square
            # R_squared > max_R_squared will keep first candidate
            # R_squared >= max_R_squared will keep last candidate
            # Keeping first or keeping last give different result, hence, this is a weakness of this method
            if keep_first:
                criteria = R_squared > max_R_squared
            else:
                criteria = R_squared >= max_R_squared

            if criteria:
                max_R_squared = R_squared
                add_feature = candidate_feature

        # Add maximum R square score feature to optimal features
        optimal_features_sets.append(add_feature)

        # Only set of features that get maximum score will be used
        score = np.mean(cross_val_score(logis_reg, x[optimal_features_sets], y, cv=10, scoring=scoring))
        # If a score is equal, select a features set that is smallest
        # So, use score > max_score will always get the smallest set of features
        if score > max_score:
            max_score = score
            max_score_features = optimal_features_sets.copy()

        # print("add_feature: {}".format(add_feature))
        # print("max_R_squared: {:.2f}%".format(max_R_squared * 100))
        # print(optimal_features_sets)
        # print("score: {:.2f}%".format(score * 100))
        # print("-----------------------------------")

    return max_score_features


def backward_selection(x, y, scoring="accuracy", keep_first=True):
    features_list = list(x.columns.values)
    iteration = len(features_list)

    optimal_features_sets = features_list.copy()
    max_score = 0
    max_score_features = None

    for _ in range(iteration):
        remove_feature = None
        max_R_squared = -1.0
        for candidate_feature in optimal_features_sets:
            # Temporally remove candidate feature from optimal set
            selected_features = [feature for feature in optimal_features_sets if feature != candidate_feature]

            # Regression y onto selected features
            logis_reg = LogisticRegression(solver="lbfgs", C=1e10, max_iter=100000)
            logis_reg.fit(x[selected_features], y)
            y_hat = logis_reg.predict(x[selected_features])

            # Calculate R square
            R_squared = R_square(y, y_hat)

            # Get maximum R square
            # R_squared > max_R_squared will keep first candidate
            # R_squared >= max_R_squared will keep last candidate
            # Keeping first or keeping last give differnt result, hence, this is a weakness of this method
            if keep_first:
                criteria = R_squared > max_R_squared
            else:
                criteria = R_squared >= max_R_squared

            if criteria:
                max_R_squared = R_squared
                remove_feature = candidate_feature

        # Remove maximum R square score feature to optimal features
        optimal_features_sets = [feature for feature in optimal_features_sets if feature != remove_feature]

        # Only set of features that get maximum score will be used
        score = np.mean(cross_val_score(logis_reg, x[optimal_features_sets], y, cv=10, scoring=scoring))
        # If a score is equal, select a features set that is smallest
        # So, use score >= max_score will always get the smallest set of features
        if score >= max_score:
            max_score = score
            max_score_features = optimal_features_sets.copy()

        # print("remove_feature: {}".format(remove_feature))
        # print("max_R_squared: {:.2f}%".format(max_R_squared * 100))
        # print(optimal_features_sets)
        # print("score: {:.2f}%".format(score * 100))
        # print("-----------------------------------")

        if len(optimal_features_sets) == 1:
            return max_score_features

    return max_score_features


def forward_backward_selection(x, y, scoring="accuracy", iteration=None, keep_first=True, report=False):
    features_list = list(x.columns.values)
    if iteration is None:
        iteration = len(features_list) * 2

    optimal_features_sets = []
    max_score = 0
    max_score_features = None

    for iter in range(iteration):
        max_R_squared = -1.0
        add_feature = None
        # Add a feature that improve model performance
        for candidate_feature in features_list:
            # Prevent duplicate features
            if candidate_feature in optimal_features_sets:
                continue

            # Define set of features to feed to regression model
            selected_features = optimal_features_sets + [candidate_feature]

            # Regression y onto selected features
            logis_reg = LogisticRegression(solver="lbfgs", C=1e10, max_iter=100000)
            logis_reg.fit(x[selected_features], y)
            y_hat = logis_reg.predict(x[selected_features])

            # Calculate R square
            R_squared = R_square(y, y_hat)

            # Get maximum R square
            # R_squared > max_R_squared will keep first candidate
            # R_squared >= max_R_squared will keep last candidate
            # Keeping first or keeping last give differnt result, hence, this is a weakness of this method
            if keep_first:
                criteria = R_squared > max_R_squared
            else:
                criteria = R_squared >= max_R_squared

            if criteria:
                max_R_squared = R_squared
                add_feature = candidate_feature

        # print("(Add) max_R_squared: {:.2f}%".format(max_R_squared * 100))
        if add_feature is not None:
            # Add best feature to optimal features
            # print("add_feature: {}".format(add_feature))
            optimal_features_sets.append(add_feature)

        remove_feature = None
        if len(optimal_features_sets) > 1:
            # Remove features that improve model performance
            for _ in range(len(optimal_features_sets)):
                remove_feature = None
                for candidate_feature in optimal_features_sets:
                    # Do not remove a just added feature
                    if candidate_feature == add_feature:
                        continue

                    # Temporally remove candidate feature from optimal set
                    selected_features = [feature for feature in optimal_features_sets if feature != candidate_feature]

                    # Regression y onto selected features
                    logis_reg = LogisticRegression(solver="lbfgs", C=1e10, max_iter=100000)
                    logis_reg.fit(x[selected_features], y)
                    y_hat = logis_reg.predict(x[selected_features])

                    # Calculate R square
                    R_squared = R_square(y, y_hat)

                    # Get maximum R square
                    if R_squared > max_R_squared:
                        max_R_squared = R_squared
                        remove_feature = candidate_feature

                if remove_feature is not None:
                    # print("(Remove) max_R_squared: {:.2f}%".format(max_R_squared * 100))
                    # Remove unnecessary feature from optimal features
                    # print("remove_feature: {}".format(remove_feature))
                    optimal_features_sets = [feature for feature in optimal_features_sets if feature != remove_feature]
                else:
                    break

        if add_feature is None and remove_feature is None:
            return max_score_features

        # Only set of features that get maximum score will be used
        score = np.mean(cross_val_score(logis_reg, x[optimal_features_sets], y, cv=10, scoring=scoring))
        # If a score is equal, select a features set that is smallest
        if score > max_score or (score >= max_score and len(optimal_features_sets) < len(max_score_features)):
            max_score = score
            max_score_features = optimal_features_sets.copy()

        if report:
            print("Completed iterations {}/{}".format(iter, iteration))
            print("max_score: {}".format(max_score))
            # print(optimal_features_sets)
            # print("score: {:.2f}%".format(score * 100))
            print("-----------------------------------")

    # print("Process is finish by maximum iteration.")
    return max_score_features


def create_polynomial(x, degree=2, interaction=False):
    assert degree > 1, "degree must be > 1, but get {}".format(degree)

    features_list = x.columns

    if interaction:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        new_x = poly.fit_transform(x)
        new_feature = poly.get_feature_names(features_list)

        polynomial = pd.DataFrame(new_x, columns=new_feature).drop(features_list, axis=1)
    else:
        polynomial = pd.DataFrame()
        new_x = []
        new_feature = []
        for feature in features_list:
            for d in range(2, degree + 1):
                new_x = np.power(x[feature], d)
                new_feature = feature + "^{}".format(d)

                polynomial[new_feature] = new_x

    return polynomial