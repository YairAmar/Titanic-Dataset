import pandas as pd
from preprocessing import PreProcessor
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import get_scorer


class ClassificationPipeline:
    """
    The classification pipeline contains the preprocessing pipeline, followed by a classifier.
    With this pipeline training, evaluating a model and applying predictions all can be done using one
    convenient object, while preventing data leakage.
    The class supports the usage of XGBoost and RandomForest classifiers.
    """
    def __init__(self, clf: str = "xgb"):
        """
        Constructs all instance attributes of the new class instance.
        """
        self.preprocessor = PreProcessor(feature_selection=False, tree_model=True)
        if clf == "xgb":
            self.classifier = XGBClassifier(learning_rate=0.01, n_estimators=600, max_depth=3, subsample=0.85,
                                            colsample_bytree=1, gamma=1, use_label_encoder=False, verbosity=0)

        else:   # assuming rfc for now...
            self.classifier = RandomForestClassifier(criterion="gini", n_estimators=1000, oob_score=True,
                                                     ccp_alpha=.0007984414423046222, min_samples_leaf=4, n_jobs=-1)

    def fit(self, data: pd.DataFrame):
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

    def predict(self, data: pd.DataFrame) -> dict:
        """
        Applies prediction over the given data.
        Args:
            data: input data to predict on.

        Returns:
            balanced_accuracy: predicted labels.
        """
        x, y = self.preprocessor.fit_transform(data, fit=False)
        y_pred = self.classifier.predict(x)
        return y_pred

    def score(self, data: pd.DataFrame, scoring_method: str = "balanced_accuracy") -> float:
        """
        Applies prediction over the given data and returns the balanced accuracy of the prediction.
        Args:
            data: input data to predict on.
            scoring_method: supports all sklearns' scoring methods, as a string.

        Returns:
            score: models' score, according to the given method.
        """
        x, y = self.preprocessor.fit_transform(data, fit=False)
        scorer = get_scorer(scoring_method)
        return scorer(estimator=self.classifier, X=x, y_true=y)
