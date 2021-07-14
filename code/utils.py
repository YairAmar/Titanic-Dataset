import pandas as pd
import argparse
import yaml


def load_data() -> tuple:
    """
    Loads a dataset, split into 2 csv files for train and test.
    Its' input is two CLI arguments: --train_path and --test_path, which are strings,
    representing the path to the training set and the test set.

    Returns:
        train: training set
        test: test set
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=False)
    args = parser.parse_args()
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    return train, test


def read_config_file() -> object:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=False)
    parser.add_argument("--test_path", type=str, required=False)
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.config_path) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return config
