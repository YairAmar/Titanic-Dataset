from utils import load_data, read_config_file
from pipeline import ClassificationPipeline


def main():
    """
    Loads the train and test set, trains a classification pipeline on the training set
    and evaluates accuracy over the test set.
    """
    train, test = load_data()
    config = read_config_file()
    classification_pipe = ClassificationPipeline(**config["main"]["classificationpipeline_params"])
    classification_pipe.fit(train)
    print(f"Balanced accuracy over train set = {classification_pipe.score(train)*100:.2f}%")
    print(f"Balanced accuracy over test set = {classification_pipe.score(test)*100:.2f}%")


if __name__ == "__main__":
    main()
