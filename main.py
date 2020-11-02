import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


from abc import ABC, abstractmethod

class Classifier(ABC):

    def __init__(self, csv="insurance_claims.csv"):
        # set missing values to Nan
        self._dataset = pd.read_csv(csv).replace('?', np.NaN)
        self._dataset.isnull().any()

        self.better_disturbute_dataset()

        self._X = self._dataset.iloc[:, :-1].values
        self._y = self._dataset.iloc[:, -1].values


        self.auto_model_vs_fraud_report = self._dataset[['auto_model']]
        self.auto_make_vs_fraud_report = self._dataset[['auto_make']]
        self.police_availaible_vs_fraud = self._dataset[['police_report_available']]
        self.property_damage_vs_fraud = self._dataset[['property_damage']]
        self.incident_city_vs_fraud = self._dataset[['incident_city']]
        self.incident_state_vs_fraud = self._dataset[['incident_state']]
        self.authorities_contacted_vs_fraud = self._dataset[['authorities_contacted']]
        self.incident_severity_vs_fraud = self._dataset[['incident_severity']]
        self.collision_type_vs_fraud = self._dataset[['collision_type']]
        self.incident_type_vs_fraud = self._dataset[['incident_type']]
        self.insured_relationship_vs_fraud = self._dataset[['insured_relationship']]
        self.insured_hobbies_vs_fraud = self._dataset[['insured_hobbies']]
        self.insured_occupation_vs_fraud = self._dataset[['insured_occupation']]
        self.insured_education_level_vs_fraud = self._dataset[['insured_education_level']]
        self.insured_sex_vs_fraud = self._dataset[['insured_sex']]
        self.policy_csl_vs_fraud = self._dataset[['policy_csl']]
        self.policy_state_vs_fraud = self._dataset[['policy_state']]

        self.preProcessing()

        # model selection
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self._X, self._y, test_size=0.2, random_state=0)

    def preProcessing(self):
        #encode the fraud reported column
        label_encoder = LabelEncoder()
        column_transfer =  ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
        self._y = label_encoder.fit_transform(self._y)

        # turn dates into time stamp
        self._X[:, 3] = pd.to_datetime(self._X[:, 3], errors='coerce')
        self._X[:, 17] = pd.to_datetime(self._X[:, 17], errors='coerce')

        # group auto_model with fraud report to find correlation between each
        self.auto_model_vs_fraud_report = self.correlation_encode(self.auto_model_vs_fraud_report, 'auto_model', 36)

        # group auto_make with fraud report to find correlation between each
        self.auto_make_vs_fraud_report = self.correlation_encode(self.auto_make_vs_fraud_report, 'auto_make', 35)

        # group police_report_available with fraud report to find correlation between each
        self.police_availaible_vs_fraud = self.correlation_encode(self.police_availaible_vs_fraud, 'police_report_available', 30)

        # group property_damage with fraud report to find correlation between each
        self.property_damage_vs_fraud = self.correlation_encode(self.property_damage_vs_fraud, 'property_damage', 27)

        # group incident_city with fraud report to find correlation between each
        self.incident_city_vs_fraud = self.correlation_encode(self.incident_city_vs_fraud, 'incident_city', 23)

        # group incident_state with fraud report to find correlation between each
        self.incident_state_vs_fraud = self.correlation_encode(self.incident_state_vs_fraud, 'incident_state', 22)

        # group authorities_contacted with fraud report to find correlation between each
        self.authorities_contacted_vs_fraud = self.correlation_encode(self.authorities_contacted_vs_fraud, 'authorities_contacted', 21)

        # group incident_severity with fraud report to find correlation between each
        self.incident_severity_vs_fraud = self.correlation_encode(self.incident_severity_vs_fraud, 'incident_severity', 20)

        # group collision_type with fraud report to find correlation between each
        self.collision_type_vs_fraud = self.correlation_encode(self.collision_type_vs_fraud, 'collision_type', 19)

        # group incident_type with fraud report to find correlation between each
        self.incident_type_vs_fraud = self.correlation_encode(self.incident_type_vs_fraud, 'incident_type', 18)

        # group insured_relationship with fraud report to find correlation between each
        self.insured_relationship_vs_fraud = self.correlation_encode(self.insured_relationship_vs_fraud, 'insured_relationship', 14)

        # group insured_hobbies with fraud report to find correlation between each
        self.insured_hobbies_vs_fraud = self.correlation_encode(self.insured_hobbies_vs_fraud, 'insured_hobbies', 13)

        # group insured_hobbies with fraud report to find correlation between each
        self.insured_occupation_vs_fraud = self.correlation_encode(self.insured_occupation_vs_fraud, 'insured_occupation', 12)

        # group insured_education_level with fraud report to find correlation between each
        self.insured_education_level_vs_fraud = self.correlation_encode(self.insured_education_level_vs_fraud, 'insured_education_level', 11)

        # group insured_sex with fraud report to find correlation between each
        self.insured_sex_vs_fraud = self.correlation_encode(self.insured_sex_vs_fraud, 'insured_sex', 10)

        # group policy_csl with fraud report to find correlation between each
        self.policy_csl_vs_fraud = self.correlation_encode(self.policy_csl_vs_fraud, 'policy_csl', 5)

        # group policy_state with fraud report to find correlation between each
        self.policy_state_vs_fraud = self.correlation_encode(self.policy_state_vs_fraud, 'policy_state', 4)

    def better_disturbute_dataset(self):
        self._dataset['incident_date'] = pd.to_datetime(self._dataset['incident_date'], errors='coerce')

        # extracting days and month from date
        self._dataset['incident_month'] = self._dataset['incident_date'].dt.month
        self._dataset['incident_day'] = self._dataset['incident_date'].dt.day
        temp = self._dataset
        self._dataset = self._dataset.drop(['fraud_reported'], axis=1)
        self._dataset['fraud_reported'] = temp['fraud_reported']

        del temp

        data = self._dataset
        needed_fraud_columns = int(0.8 * len(data[data['fraud_reported'] == 'Y']))
        data_fraud = data[data['fraud_reported'] == 'Y'].head(needed_fraud_columns)
        data_not_fraud = data[data['fraud_reported'] == 'N'].head(needed_fraud_columns)

        self._dataset  = pd.concat([data_fraud, data_not_fraud])

    def correlation_encode(self, model_vs_fraud, model, index):
        model_vs_fraud = model_vs_fraud.join(pd.DataFrame(self._y, columns=['fraud_reported']))
        model_vs_fraud = model_vs_fraud.groupby([model],
                as_index = False).mean().sort_values(by ='fraud_reported', ascending = True)

        # encode model to its corresponding relevance
        fraud_mean = [round(mean, 3) for mean in model_vs_fraud['fraud_reported']]
        self._dataset['auto_model'] = self._dataset[model].replace(tuple(model_vs_fraud[model]), fraud_mean)
        self._X[:, index] = self._dataset['auto_model']

        return model_vs_fraud


    def save_plots(self):
        pass

    def correlation_heat_map(self):
        x = pd.DataFrame(self.x_train, columns=list(clf.dataset.columns)[:-1])

        x = x.drop(['incident_date', 'policy_bind_date', 'incident_location'], axis=1)

        plt.figure(figsize=(15, 12))
        plt.title("correlation matrix")
        sns.set_context('paper', font_scale=1.4)
        sns.heatmap(x.astype(float).corr(), annot=False, cmap='Blues')
        plt.savefig("plots/correlation_matrix.png")
        plt.clf()

    def info(self):
        self._dataset.info()

    def describe(self):
        return self._dataset.describe()

    def corr(self):
        return self._dataset.corr()

    @abstractmethod
    def fit(self):
        pass

    @property
    def X(self):
        return self._X
    @property
    def y(self):
        return self._y

    @property
    def dataset(self):
        return self._dataset

    @X.setter
    def X(self, val):
        self._X = val

    @y.setter
    def y(self, val):
        self._y = val


class Logistic(Classifier):

    def fit(self):
        pass


if __name__ == "__main__":
    from collections import Counter
    clf = Logistic()
    count = Counter(clf.y)
    print(count)
    print(clf.auto_model_vs_fraud_report)
    print(clf.auto_make_vs_fraud_report)
    print(clf.police_availaible_vs_fraud)
    print(clf.property_damage_vs_fraud)
    print(clf.incident_city_vs_fraud)
    print(clf.incident_state_vs_fraud)
    print(clf.authorities_contacted_vs_fraud)
    print(clf.incident_severity_vs_fraud)
    print(clf.collision_type_vs_fraud)
    print(clf.incident_type_vs_fraud)
    print(clf.insured_relationship_vs_fraud)
    print(clf.insured_hobbies_vs_fraud)
    print(clf.insured_occupation_vs_fraud)
    print(clf.insured_education_level_vs_fraud)
    print(clf.insured_sex_vs_fraud)
    print(clf.policy_csl_vs_fraud)
    print(clf.policy_state_vs_fraud)

    # print(clf.X[:, 18:24])
    # print(clf.X[:, 10:15])
    # print(clf.X[:, 4:6])

    x = pd.DataFrame(clf.x_train, columns=list(clf.dataset.columns)[:-1])

    x = x.drop(['incident_date', 'policy_bind_date', 'incident_location'], axis=1)

    # x.astype(float).corr().to_csv("x_train_corr.csv", ",")

    print(x)

    clf.correlation_heat_map()

