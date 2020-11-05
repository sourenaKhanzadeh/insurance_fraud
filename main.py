import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from abc import ABC, abstractmethod

class Classifier(ABC):
    classifier_name = "[ENTER NAME]"

    def __init__(self, csv="insurance_claims.csv", all_data=False):
        # set missing values to Nan
        self._dataset = pd.read_csv(csv).replace('?', np.NaN)
        self._dataset.isnull().any()
        self._dataset.fillna(0)

        self._all_data = all_data
        self._clf = None


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
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def preProcessing(self):
        #encode the fraud reported column
        label_encoder = LabelEncoder()
        column_transfer =  ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

        self._y = self._y.astype(str)
        self._y = label_encoder.fit_transform(self._y)

        # turn dates into time stamp
        # self._X[:, 3] = pd.to_datetime(self._X[:, 3], errors='coerce')
        # self._X[:, 17] = pd.to_datetime(self._X[:, 17], errors='coerce')
        # self._X = np.delete(self._X, 3, 1)
        # self._X = np.delete(self._X, 16, 1)

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

        if not self._all_data:
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
        self.correlation_heat_map()
        self.sex_ratio_pie_chart()
        self.scatter_matrix()
        self.property_claim_damage()
        self.incident_collision_type_severity()
        self.graph_witnesses_time_loc()
        self.graph_hobbies()

    def correlation_heat_map(self):
        # model selection
        x_train, x_test, y_train, y_test = train_test_split(self._X, self._y, test_size=0.2,
                                                                                random_state=0)

        x_train = np.column_stack((x_train, y_train))

        x = pd.DataFrame(x_train, columns=list(self._dataset.columns))

        x = x.drop(['incident_date', 'policy_bind_date', 'incident_location'], axis=1)

        x.astype(float).corr().to_csv("x_train_corr.csv")

        plt.figure(figsize=(15, 12))
        plt.title("correlation matrix")
        sns.set_context('paper', font_scale=1.4)
        sns.heatmap(x.astype(float).corr(), annot=False, cmap='Blues')
        plt.savefig("plots/correlation_matrix.png")
        plt.clf()
        plt.close()

    def sex_ratio_pie_chart(self):

        labels = "Male", "Female"
        sizes = [len(self.dataset[self.dataset['insured_sex'] == 'MALE']),len(self.dataset[self.dataset['insured_sex'] == "FEMALE"])]
        explode  = (0, 0.1)

        plt.figure(figsize=(8, 8))
        plt.title("Male Vs Female Ratio")
        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        plt.savefig('plots/sex_ratio.png')
        plt.clf()
        plt.close()

    def scatter_matrix(self):

        plt.figure(figsize=(10,10))
        sns.set_theme(style="ticks")

        data = pd.concat([self.dataset["insured_sex"],
                          self.dataset["injury_claim"],
                          self.dataset["vehicle_claim"],
                          self.dataset["property_claim"]], axis=1)

        sns.pairplot(data, hue="insured_sex")
        plt.savefig("plots/claims_matrix.png")
        plt.clf()
        plt.close()

    def property_claim_damage(self):
        plt.figure(figsize=(10, 10))
        sns.stripplot(x=self.dataset['property_damage'], y=self.dataset['property_claim'], hue=self.dataset['fraud_reported'])
        plt.savefig("plots/property_claim_damage.png")
        plt.clf()
        plt.close()

    def incident_collision_type_severity(self):
        g = sns.FacetGrid(self.dataset, col="incident_type", hue="incident_severity")
        g.map(sns.scatterplot, "capital-gains", "collision_type")
        g.add_legend()
        plt.savefig("plots/incident_collision_type_severity")
        plt.clf()
        plt.close()
        g = sns.FacetGrid(self.dataset, col="insured_sex", hue="fraud_reported", height=7)
        g.map(sns.countplot, 'incident_type', order=["Single Vehicle Collision", "Vehicle Theft", "Multi-vehicle Collision", "Parked Car"])
        g.add_legend()
        plt.savefig("plots/number_of_frauds_incident_type_by_sex.png")
        plt.clf()
        plt.close()

    def graph_witnesses_time_loc(self):
        g = sns.FacetGrid(self.dataset, col="police_report_available", hue="fraud_reported", height=5)
        g.map(sns.barplot, "witnesses", "number_of_vehicles_involved")
        # sns.kdeplot(data=self.dataset, x="witnesses", y="number_of_vehicles_involved", hue="police_report_available")
        g.add_legend()
        plt.savefig("plots/graph_witness_no_vehicle.png")
        plt.clf()
        plt.close()

    def graph_hobbies(self):
        plt.figure(figsize=(20, 12))
        sns.countplot(data=self.dataset, x="insured_hobbies", hue="fraud_reported", palette="Set3")
        plt.savefig("plots/hobbies.png")
        plt.clf()
        plt.close()
        plt.figure(figsize=(20, 12))
        sns.countplot(data=self.dataset, x="insured_occupation", hue="fraud_reported", palette="pastel")
        plt.savefig("plots/occupation.png")
        plt.clf()
        plt.close()
        sns.countplot(data=self.dataset, x="insured_education_level", hue="fraud_reported", palette="cool")
        plt.savefig("plots/education_level.png")
        plt.clf()
        plt.close()

    def info(self):
        self._dataset.info()

    def describe(self):
        return self._dataset.describe()

    def corr(self):
        return self._dataset.corr()

    @abstractmethod
    def fit(self):
        self._X = np.delete(self._X, 3, 1)
        self._X = np.delete(self._X, 16, 1)
        self._X = np.delete(self._X, 22, 1)

        # model selection
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self._X, self._y, test_size=0.2, random_state=0)

        self.x_train = pd.DataFrame(self.x_train).fillna(0)

        self.x_test = pd.DataFrame(self.x_test).fillna(0)

        self.x_train = self.x_train.to_numpy()

        self.x_test= self.x_test.to_numpy()

    @abstractmethod
    def accuracy(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def confusion_matrix(self):
        pass

    def draw_confusion_matrix(self):
        sns.heatmap(self.confusion_matrix(), annot=True, cmap='spring')
        plt.show()

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


    def __str__(self):
        res = """
    ===========================
    {} Classifier
    ===========================
    Confusion Matrix: {}
    Training Accuracy: {}
    Testing Accuracy: {}
        """.format(
            self.classifier_name,
            np.array(self.confusion_matrix()),
            self._clf.score(self.x_train, self.y_train),
            self._clf.score(self.x_test, self.y_test)
        )

        return res

class Logistic(Classifier):
    classifier_name = "Logistic"


    def __init__(self, csv="insurance_claims.csv", all_data=False):
        super().__init__(csv, all_data)


    def fit(self):
        super().fit()

        clf = LogisticRegression(random_state=0)
        clf.fit(self.x_train, self.y_train)

        self._clf = clf

    def accuracy(self):
        return accuracy_score(self.y_test, self.predict())

    def predict(self):
        return self._clf.predict(self.x_test)

    def confusion_matrix(self):
        return confusion_matrix(self.y_test, self.predict())

    def __str__(self):
        res = super().__str__()
        res += """
    Accuracy: {}
    {}
        """.format(
            self.accuracy(),
            classification_report(self.y_test, self.predict())
        )

        return res

class SVC_(Classifier):

    classifier_name = "SVM"

    def fit(self):
        super().fit()

        clf = SVC(kernel='sigmoid', random_state=0)
        clf.fit(self.x_train, self.y_train)

        self._clf = clf

    def accuracy(self):
        return accuracy_score(self.y_test, self.predict())

    def predict(self):
        return self._clf.predict(self.x_test)

    def confusion_matrix(self):
        return confusion_matrix(self.y_test, self.predict())

    def __str__(self):
        res = super().__str__()
        res += """
    Accuracy: {}
    {}
        """.format(
            self.accuracy(),
            classification_report(self.y_test, self.predict())
        )

        return res

class KNN(Classifier):
    classifier_name = "K-NN"

    def fit(self):
        super().fit()

        clf = KNeighborsClassifier(n_neighbors=15, metric="minkowski", p=1)
        clf.fit(self.x_train, self.y_train)

        self._clf = clf

    def accuracy(self):
        return accuracy_score(self.y_test, self.predict())

    def predict(self):
        return self._clf.predict(self.x_test)

    def confusion_matrix(self):
        return confusion_matrix(self.y_test, self.predict())

    def __str__(self):
        res = super().__str__()
        res += """
       Accuracy: {}
       {}
           """.format(
            self.accuracy(),
            classification_report(self.y_test, self.predict())
        )

        return res

class NaiveBayes(Classifier):
    classifier_name = "Naive Bayes"

    def fit(self):
        super().fit()

        clf = GaussianNB()
        clf.fit(self.x_train, self.y_train)

        self._clf = clf

    def accuracy(self):
        return accuracy_score(self.y_test, self.predict())

    def predict(self):
        return self._clf.predict(self.x_test)

    def confusion_matrix(self):
        return confusion_matrix(self.y_test, self.predict())

    def __str__(self):
        res = super().__str__()
        res += """
       Accuracy: {}
       {}
           """.format(
            self.accuracy(),
            classification_report(self.y_test, self.predict())
        )

        return res


class DecisionTree(Classifier):
    classifier_name = "Decision Tree"

    def fit(self):
        super().fit()

        clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
        clf.fit(self.x_train, self.y_train)

        self._clf = clf

    def accuracy(self):
        return accuracy_score(self.y_test, self.predict())

    def predict(self):
        return self._clf.predict(self.x_test)

    def confusion_matrix(self):
        return confusion_matrix(self.y_test, self.predict())

    def __str__(self):
        res = super().__str__()
        res += """
       Accuracy: {}
       {}
           """.format(
            self.accuracy(),
            classification_report(self.y_test, self.predict())
        )

        return res


class RandomForest(Classifier):
    classifier_name = "Random Forest"

    def fit(self):
        super().fit()

        clf = RandomForestClassifier(n_estimators=10,criterion="entropy", random_state=0)
        clf.fit(self.x_train, self.y_train)

        self._clf = clf

    def accuracy(self):
        return accuracy_score(self.y_test, self.predict())

    def predict(self):
        return self._clf.predict(self.x_test)

    def confusion_matrix(self):
        return confusion_matrix(self.y_test, self.predict())

    def __str__(self):
        res = super().__str__()
        res += """
           Accuracy: {}
           {}
               """.format(
            self.accuracy(),
            classification_report(self.y_test, self.predict())
        )

        return res


if __name__ == "__main__":
    from collections import Counter
    from warnings import filterwarnings

    filterwarnings("ignore")

    clf = Logistic(all_data=False)

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


    # clf.save_plots()


    # fit the model
    clf.fit()

    # get logistic classifier details
    print(clf)

    # Support Vector Machine
    svm = SVC_(all_data=False)

    # fit the model
    svm.fit()

    print(svm)

    # K nearest neighbors
    knn = KNN(all_data=False)

    # fit the model
    knn.fit()

    print(knn)

    # Naive Bayes
    nb = NaiveBayes(all_data=False)

    # fit model
    nb.fit()

    print(nb)

    # Decision Tree
    dt = DecisionTree(all_data=False)

    # fit the model
    dt.fit()

    print(dt)

    #Random Forest
    rf = RandomForest(all_data=False)

    #fit the model
    rf.fit()


    print(rf)








