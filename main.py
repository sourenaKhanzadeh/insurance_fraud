import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer


from abc import ABC, abstractmethod

class Classifier(ABC):

    def __init__(self, csv="insurance_claims.csv"):
        # set missing values to Nan
        self._dataset = pd.read_csv(csv).replace('?', np.NaN)
        self._dataset.isnull().any()


        self._X = self._dataset.iloc[:, :-1].values
        self._y = self._dataset.iloc[:, -1].values


        self.auto_model_vs_fraud_report = self._dataset[['auto_model']]
        self.auto_make_vs_fraud_report = self._dataset[['auto_make']]

        self.preProcessing()


    def preProcessing(self):
        #encode the fraud reported column
        label_encoder = LabelEncoder()
        column_transfer =  ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

        self._y = label_encoder.fit_transform(self._y)

        # set policy_bind_date to time spam
        self._X[:, 2] = pd.to_datetime(self._X[:, 2], errors='coerce')

        # group auto_model with fraud report to find correlation between each
        self.auto_model_vs_fraud_report = self.correlation_encode(self.auto_model_vs_fraud_report, 'auto_model', 36)

        # group auto_make with fraud report to find correlation between each
        self.auto_make_vs_fraud_report = self.correlation_encode(self.auto_make_vs_fraud_report, 'auto_make', 35)

    def correlation_encode(self, model_vs_fraud, model, index):
        model_vs_fraud = model_vs_fraud.join(pd.DataFrame(self._y, columns=['fraud_reported']))
        model_vs_fraud = model_vs_fraud.groupby([model],
                as_index = False).mean().sort_values(by ='fraud_reported', ascending = True)

        # encode model to its corresponding relevance
        fraud_mean = [round(mean, 3) for mean in model_vs_fraud['fraud_reported']]
        self._dataset['auto_model'] = self._dataset[model].replace(tuple(model_vs_fraud[model]), fraud_mean)
        self._X[:, index] = self._dataset['auto_model']

        return model_vs_fraud




    def info(self):
        self._dataset.info()

    def describe(self):
        return self._dataset.describe()


    @property
    def X(self):
        return self._X
    @property
    def y(self):
        return self._y

    @X.setter
    def X(self, val):
        self._X = val

    @y.setter
    def y(self, val):
        self._y = val


class Logistic(Classifier):

    pass


if __name__ == "__main__":
    clf = Logistic()
    print(clf.auto_model_vs_fraud_report)
    print(clf.auto_make_vs_fraud_report)
    print(clf.X[:, 35:37])

