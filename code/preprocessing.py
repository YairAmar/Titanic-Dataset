import numpy as np
import pandas as pd
import re
from utils import read_config_file
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class PreProcessor:
    """
    The PreProcessor is an transformer that applies all the needed steps in order to prepare the
    Titanic dataset to classification, while implementing an interface similar to the one in sklearns' transformers.
    """
    def __init__(self, feature_selection: bool = True, tree_model: bool = True):
        """
        Constructs all instance attributes of the new class instance.
        """
        config = read_config_file()
        if not tree_model:
            self.scaler = StandardScaler()

        self.tree_model = tree_model
        self.encoder = OneHotEncoder(**config["preprocessing"]["onehotdncoder_params"])
        if feature_selection:
            self.feature_selection_clf = LogisticRegression(**config["preprocessing"]["feature_selection_clf_params"])
            self.selector = RFECV(self.feature_selection_clf, **config["preprocessing"]["RFECV_params"])
            self.feature_selection = True
        else:
            self.feature_selection = False

    def fit_transform(self, data: pd.DataFrame, fit: bool = True) -> tuple:
        """
        Fits the PreProcessor object (transformer) to the data, by constructing and fitting it's instance variables.
        After fitting it applies the whole pre-processing pipeline over the data.

        Args:
            data: input data
            fit: whether to fit the transformers or only transform (default True).

        Returns:
            x: data ready to feed a classification model
            y: labels of the data
        """
        data_copy = data.copy()
        data_copy = self._clean_data(data_copy)
        data_copy = self._insert_features(data_copy, fit)
        # split the data into x and y
        config = read_config_file()
        x = data_copy.drop(config["preprocessing"]["y_column"], axis=1)
        y = data_copy[config["preprocessing"]["y_column"]]

        if not self.tree_model:
            if fit:
                self.scaler.fit(x.loc[:, config["preprocessing"]["continuous_features"]])

            x.loc[:, config["preprocessing"]["continuous_features"]] = self.scaler.transform(
                x.loc[:, config["preprocessing"]["continuous_features"]], copy=False)

        if self.feature_selection:
            x, y = self._feature_selection(x, y, fit=fit)
        return x, y

    @staticmethod
    def _clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleaning the input data
         - removing "Cabin" and "Ticket" columns
         - removing NaN values from "Embarked" column
        Args:
            data: input data

        Returns:
            data: clean version of the input data
        """
        config = read_config_file()
        data.drop(config["preprocessing"]["drop_parameters"], axis=1, inplace=True)
        data.dropna(subset=config["preprocessing"]["dropna_parameters"], inplace=True)
        data.reset_index(inplace=True)
        return data

    def _insert_age_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Turning the contiguous feature "Age" into a categorical feature and into one-hot-vector
        Args:
            data: input data
        Returns:
            data: input data with Age grouped into one-hot-vector
        """
        data["Age"].fillna(-1, inplace=True)  # unknown age = -1
        if self.tree_model is False:
            category = pd.cut(data.Age, bins=[-2, 0, 2, 17, 30, 65, 100],
                              labels=['Unknown', 'Toddler/baby', 'Child', 'Young_adult', 'Adult', 'Elderly'])
            data.insert(6, 'Age_Group', category)
            data['Age_Unknown'] = (data['Age_Group'] == 'Unknown').astype(int)
            data['Age_Toddler'] = (data['Age_Group'] == 'Toddler/baby').astype(int)
            data['Age_Child'] = (data['Age_Group'] == 'Child').astype(int)
            data['Age_Young_adult'] = (data['Age_Group'] == 'Young_adult').astype(int)
            data['Age_Adult'] = (data['Age_Group'] == 'Adult').astype(int)
            data['Age_Elderly'] = (data['Age_Group'] == 'Elderly').astype(int)
            data.drop(["Age"], axis=1, inplace=True)
            data.drop(["Age_Group"], axis=1, inplace=True)
        return data

    @staticmethod
    def _insert_fare_feature(data: pd.DataFrame) -> pd.DataFrame:
        """
        Turning the contiguous feature "Fare" into a categorical feature and into one-hot-vector
        Args:
            data: input data
        Returns:
            data: input data with Fare grouped into one-hot-vector
        """
        category = pd.cut(data.Fare, bins=[-0.1, 0.1, 25, 50, 100, 1000], labels=[0, 1, 2, 3, 4])
        data.insert(5, 'Fare_Category', category)
        data['Fare_Category_0'] = (data['Fare_Category'] == 0).astype(int)
        data['Fare_Category_1'] = (data['Fare_Category'] == 1).astype(int)
        data['Fare_Category_2'] = (data['Fare_Category'] == 2).astype(int)
        data['Fare_Category_3'] = (data['Fare_Category'] == 3).astype(int)
        data['Fare_Category_4'] = (data['Fare_Category'] == 4).astype(int)
        data.drop(["Fare"], axis=1, inplace=True)
        data.drop(["Fare_Category"], axis=1, inplace=True)
        return data

    @staticmethod
    def _insert_family_size_feature(data: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs a new feature, Family_Size=Parch+SibSp
        Args:
            data: input data

        Returns:
            data: input data, with Family_size feature
        """
        data["Family_Size"] = data["Parch"] + data["SibSp"]
        return data

    @staticmethod
    def _insert_single_feature(data: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs a new feature, indicating whether the passenger is single
        Args:
            data: input data

        Returns:
            data: input data, with Single feature
        """
        data["Single"] = np.logical_and(data["Parch"] == 0, data["SibSp"] == 0).astype(int)
        return data

    @staticmethod
    def _insert_lot_sib_feature(data: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs a new feature, indicating whether the passenger has more than 2 siblings
        Args:
            data: input data

        Returns:
            data: input data, with Lot_Sib feature
        """
        data["Lot_Sib"] = (data["SibSp"] >= 2).astype(int)
        return data

    @staticmethod
    def _insert_lot_child_feature(data: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs a new feature, indicating whether the passenger has more than 4 parents+children aboard
        Args:
            data: input data

        Returns:
            data: input data, with Lot_Child feature
        """
        data["Lot_Child"] = (data["Parch"] >= 4).astype(int)
        return data

    @staticmethod
    def _insert_big_fam_feature(data: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs a new feature, indicating whether the passenger has more than 4 family members aboard
        Args:
            data: input data

        Returns:
            data: input data, with Big_Fam feature
        """
        data["Big_Fam"] = (data["Family_Size"] >= 4).astype(int)
        return data

    @staticmethod
    def _insert_mid_class_q_feature(data: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs a new feature, indicating whether the passenger is mid-class and embarked in Queenstown.
        Args:
            data: input data

        Returns:
            data: input data, with Mid_Class_Q feature
        """
        data["Mid_Class_Q"] = np.logical_and(data["Pclass"] == 2, data["Embarked"] == 2).astype(int)
        return data

    @staticmethod
    def _insert_title_feature(data: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs a new feature, extracting the passenger's title from the name.
        Args:
            data: input data.

        Returns:
            data: input data, with Title feature.
        """
        data["Title"] = data["Name"].apply(lambda var: re.split("\.", re.split(",", var)[1])[0])
        data.drop(["Name"], axis=1, inplace=True)
        return data

    def _dummy_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Turns Sex,Embarked and Title features into one-hot-vectors.
        In the way of doing so it fits the OneHotEncoder of the class's instance.
        Args:
            data: input data
            fit: whether to fit the OneHotEncoder or just transform (default True)

        Returns:
            data: input data, with relevant features transformed.
        """
        data["Sex"].replace({"female": 0, "male": 1}, inplace=True)

        if fit:
            self.encoder.fit(data[["Embarked", "Title"]])

        one_hot_features = self.encoder.transform(data[["Embarked", "Title"]])
        column_names = self.encoder.get_feature_names(["Embarked", "Title"])
        one_hot_features = pd.DataFrame(one_hot_features, columns=column_names)
        data = data.join(one_hot_features)
        data.drop(labels=["Embarked", "Title"], axis=1, inplace=True)
        return data

    def _insert_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Creates all the features needed in this class, while also training the fitting transformers.
        Args:
            data: input data
            fit: whether to fit the transformers or just transform (default True)

        Returns:
            data: input data, after all feature manipulations applied in the preprocessing stage.
        """
        data = self._insert_age_feature(data)
        data = self._insert_family_size_feature(data)

        if not self.tree_model:
            data = self._insert_fare_feature(data)
            data = self._insert_single_feature(data)
            data = self._insert_lot_sib_feature(data)
            data = self._insert_lot_child_feature(data)
            data = self._insert_big_fam_feature(data)
            data = self._insert_mid_class_q_feature(data)

        data = self._insert_title_feature(data)
        data = self._dummy_features(data, fit)
        return data

    def _feature_selection(self, x: pd.DataFrame, y: pd.Series, fit: bool = True) -> tuple:
        """
        Selects the features most helpful in classifying the data, by iteratively adding them 1 by 1,
        and checking accuracy over 5-fold-CV.
        Args:
            x: input data.
            y: target values for the classification task.
            fit: whether to fit the transformers or just transform (default True).

        Returns:
            x: data samples, in reduced feature vectors.
        """
        if fit:
            self.selector.fit(x, y)

        cols = self.selector.get_support(indices=True)
        x = x.iloc[:, cols]
        return x, y
