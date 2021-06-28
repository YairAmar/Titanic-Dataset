import pandas as pd


def load_data(path_train: str, path_test: str) -> tuple:
    """
    Loads a dataset, split into 2 csv files for train and test
    Args:
        path_train: absolute path of training set
        path_test: absolute path of test set

    Returns:
        train: training set
        test: test set
    """
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)

    return train, test
