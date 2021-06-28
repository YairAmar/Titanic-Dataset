import pandas as pd
from preprocessing import PreProcessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score


class ClassificationPipeline:
    """
    The classification pipeline contains the preprocessing pipeline, followed by a classifier.
    With this pipeline training, evaluating a model and applying predictions all can be done using one
    convenient object, while preventing data leakage.
    """
    def __init__(self):
        """
        Constructs all instance attributes of the new class instance.
        """
        self.preprocessor = PreProcessor(feature_selection=False)
        self.classifier = RandomForestClassifier(criterion="gini",
                                                 n_estimators=10000,
                                                 oob_score=True,
                                                 ccp_alpha=.0007984414423046222,
                                                 min_samples_leaf=3,
                                                 n_jobs=-1)

    def fit(self, data: pd.DataFrame) -> dict:
        """
        Fits the preprocessing and classification model over the given data.
        Args:
            data: input data to fit the model on.

        Returns:
            train_balanced_accuracy: balanced accuracy over the training set.
            out_of_bag_accuracy: out of bag accuracy of the RandomForest model.
        """
        x, y = self.preprocessor.fit_transform(data)
        self.classifier.fit(x, y)
        y_pred = self.classifier.predict(x)
        balanced_accuracy = balanced_accuracy_score(y, y_pred)
        return {"train_balanced_accuracy": balanced_accuracy,
                "out_of_bag_accuracy": self.classifier.oob_score}

    def predict(self, data: pd.DataFrame) -> dict:
        """
        Applies prediction over the given data.
        Args:
            data: input data to predict on.

        Returns:
            balanced_accuracy: predicted labels.
        """
        x, y = self.preprocessor.transform(data)
        y_pred = self.classifier.predict(x)
        return y_pred

    def balanced_score(self, data: pd.DataFrame) -> float:
        """
        Applies prediction over the given data and returns the balanced accuracy of the prediction.
        Args:
            data: input data to predict on.

        Returns:
            balanced_accuracy: balanced accuracy of the model in predicting over the given data.
        """
        x, y = self.preprocessor.transform(data)
        y_pred = self.classifier.predict(x)
        balanced_accuracy = balanced_accuracy_score(y, y_pred)
        return balanced_accuracy
