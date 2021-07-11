import sys
from utils import load_data
from pipeline import ClassificationPipeline


def main():
    """
    Loads the train and test set, trains a classification pipeline on the training set
    and evaluates accuracy over the test set.
    """
    train, test = load_data(sys.argv[1], sys.argv[2])
    classification_pipe = ClassificationPipeline()
    train_score = classification_pipe.fit(train)
    print(f"The train score of a XGB without feature selection is: "
          f"{train_score:.3f}")
    print(f"The test score of a XGB without feature selection is: "
          f"{classification_pipe.balanced_score(test):.3f}")


if __name__ == "__main__":
    main()
