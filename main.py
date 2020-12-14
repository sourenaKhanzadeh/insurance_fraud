import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from mpl_toolkits.mplot3d import Axes3D

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
            needed_fraud_columns = int(1 * len(data[data['fraud_reported'] == 'Y']))
            # data_fraud = data[data['fraud_reported'] == 'Y'].head(needed_fraud_columns)
            # data_not_fraud = data[data['fraud_reported'] == 'N'].head(needed_fraud_columns + 24)
            data_fraud = data[data['fraud_reported'] == 'Y'].head(230)
            data_not_fraud = data[data['fraud_reported'] == 'N'].head(230)

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

    def cross_valid(self):
        kf = StratifiedKFold(n_splits=5)
        scores = []

        for train_i, test_i in kf.split(self._X, self._y):
            x_train, x_test, y_train, y_test = self._X[train_i], self._X[test_i], self._y[train_i], self._y[test_i]

            x_train = pd.DataFrame(x_train).fillna(0)
            x_test = pd.DataFrame(x_test).fillna(0)
            x_train = x_train.to_numpy()
            x_test = x_test.to_numpy()

            self._clf.fit(x_train, y_train)
            scores.append(self._clf.score(x_test, y_test))

        return np.mean(np.array(scores))


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
    par ="_Sag_penalty_none"


    def __init__(self, csv="insurance_claims.csv", all_data=False):
        super().__init__(csv, all_data)


    def fit(self):
        super().fit()

        clf = LogisticRegression(random_state=0, solver="newton-cg", penalty='none')
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
    par = "Poly"

    def fit(self):
        super().fit()

        clf = SVC(kernel='poly', random_state=0)
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
    par = "33_Man"

    def fit(self):
        super().fit()

        clf = KNeighborsClassifier(n_neighbors=33, metric="minkowski", p=2)
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

    def test_neigboursize_manhattan(self):
        super().fit()
        x = []
        y = []
        for i in range(1, 50):
            x.append(i)

            clf = KNeighborsClassifier(n_neighbors=i, metric="minkowski", p=2)
            clf.fit(self.x_train, self.y_train)

            self._clf = clf

            y.append(self.cross_valid())

        plt.title("Num_Neighbours")
        plt.xlabel("num neighbours")
        plt.ylabel("cross_validation")

        plt.plot(x, y, color='red')

        index = y.index(max(y))

        print("n_neigbors={}, cross_valid={}".format(x[index], y[index]))

        plt.savefig("plots/Num_neighbours_K_NN.png")


class DecisionTree(Classifier):
    classifier_name = "Decision Tree"
    par = ""

    def fit(self):
        super().fit()

        clf = DecisionTreeClassifier(criterion="entropy", random_state=0, min_samples_leaf=18)
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

    def test_leaves(self):
        super().fit()
        leaf_cross_valid = []
        for i in range(1, 51):
            clf = DecisionTreeClassifier(criterion="entropy", random_state=0, min_samples_leaf=i)
            clf.fit(self.x_train, self.y_train)
            self._clf = clf
            leaf_cross_valid.append(self.cross_valid())
        return leaf_cross_valid


class RandomForest(Classifier):
    classifier_name = "Random Forest"
    par = "_Tree__83_leaf_size__1"

    def fit(self):
        super().fit()

        clf = RandomForestClassifier(n_estimators=83,criterion="entropy", random_state=1, min_samples_leaf=1, max_features=25)
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

    def forest_optim_graph(self, graph=True):
        super().fit()
        if not graph:

            maxima = []
            maxima_i = []
            for k in  range(1, 26, 2):
                x = []
                y = []
                z = []
                for i in range(1, 101):
                    for j in range(k, k+2):

                        clf = RandomForestClassifier(n_estimators=i, criterion="entropy", random_state=1, min_samples_leaf=j,
                                                     max_features=25)
                        clf.fit(self.x_train, self.y_train)
                        self._clf = clf

                        x.append(i)
                        y.append(j)
                        z.append(self.cross_valid())
                index = z.index(max(z))
                print("({}, {}): tree_size = {}, leaf_size = {} , cross_validation = {}".format(k, k+5, x[index], y[index], z[index]))
                maxima.append([x[index], y[index], z[index]])

            index = 0
            for i in range(len(maxima)):
                if maxima[i][-1] > maxima[index][-1]:
                    index = i

            print("Maxima: tree_size = {}, leaf_size = {} , cross_validation = {}".format(maxima[index][0], maxima[index][1], maxima[index][-1]))

        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # modify x,y,z according to the report

            x = [83, 58, 72, 54, 20, 2, 2, 16, 18, 29, 24, 38, 7]
            y = [1, 4, 5, 7, 9, 12, 13, 15, 17, 19, 21, 23, 25]
            z = []
            for i, j in zip(x, y):
                clf = RandomForestClassifier(n_estimators=i, criterion="entropy", random_state=1, min_samples_leaf=j,
                                             max_features=25)
                clf.fit(self.x_train, self.y_train)
                self._clf = clf

                z.append(self.cross_valid())

            ax.set_xlabel("tree_size")
            ax.set_ylabel("leaf_size")
            ax.set_zlabel("cross_validation")
            ax.scatter(x, y, z)

            plt.show()


class NN(Classifier):
    classifier_name = "Neural Network MLP"
    par = "NN_92_92_tanh_adam"

    def fit(self):
        super().fit()

        clf = MLPClassifier(activation="tanh",solver='adam',
                     hidden_layer_sizes=(92, 92), random_state=1)

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

    def test_hidden_layers(self):
        super().fit()

        x = []
        y = []
        z = []
        h = []
        for active in ['identity', 'logistic', 'tanh', 'relu']:
            for solver in ['adam', 'sgd', 'lbfgs']:
                for i in range(37, 100):
                    clf = MLPClassifier(activation=active, solver=solver,
                                        hidden_layer_sizes=(i, i), random_state=1)

                    clf.fit(self.x_train, self.y_train)

                    self._clf = clf

                    x.append(i)
                    y.append(self.cross_valid())
                    z.append(active)
                    h.append(solver)

        d = pd.DataFrame(zip(x, y, z, h), columns=['num_hidden_layers', 'cross_valid', 'activation', 'solver'])
        g = sns.FacetGrid(d, col="activation",height=20, hue='solver')
        g.map(sns.pointplot, "num_hidden_layers", "cross_valid")
        # sns.kdeplot(data=self.dataset, x="witnesses", y="number_of_vehicles_involved", hue="police_report_available")
        g.add_legend()
        plt.savefig("plots/optimal_NN.png")
        plt.clf()
        plt.close()

class Ada(Classifier):
    classifier_name = "AdaBoost Classifier: "
    par = ""

    def __init__(self, base_clf, csv="insurance_claims.csv", all_data=False):
        super().__init__(csv, all_data)
        self.base_clf = base_clf


        self.classifier_name += base_clf.classifier_name

    def fit(self):
        super().fit()

        clf = AdaBoostClassifier(base_estimator=self.base_clf._clf)

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


class Bag(Classifier):
    classifier_name = "Bagging Classifier: "
    par = ""

    def __init__(self, base_clf, csv="insurance_claims.csv", all_data=False):
        super().__init__(csv, all_data)
        self.base_clf = base_clf

        self.classifier_name += base_clf.classifier_name

    def fit(self):
        super().fit()

        clf = BaggingClassifier(base_estimator=self.base_clf._clf)

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

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 10))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")



    # Plot n_samples vs fit_times
    # axes[1].grid()
    # axes[1].plot(train_sizes, fit_times_mean, 'o-')
    # axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
    #                      fit_times_mean + fit_times_std, alpha=0.1)
    # axes[1].set_xlabel("Training examples")
    # axes[1].set_ylabel("fit_times")
    # axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    # axes[2].grid()
    # axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    # axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1)
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")

    return plt

if __name__ == "__main__":
    from collections import Counter
    from warnings import filterwarnings

    filterwarnings("ignore")

    lg = Logistic(all_data=False)

    count = Counter(lg.y)
    print(count)
    print(lg.auto_model_vs_fraud_report)
    print(lg.auto_make_vs_fraud_report)
    print(lg.police_availaible_vs_fraud)
    print(lg.property_damage_vs_fraud)
    print(lg.incident_city_vs_fraud)
    print(lg.incident_state_vs_fraud)
    print(lg.authorities_contacted_vs_fraud)
    print(lg.incident_severity_vs_fraud)
    print(lg.collision_type_vs_fraud)
    print(lg.incident_type_vs_fraud)
    print(lg.insured_relationship_vs_fraud)
    print(lg.insured_hobbies_vs_fraud)
    print(lg.insured_occupation_vs_fraud)
    print(lg.insured_education_level_vs_fraud)
    print(lg.insured_sex_vs_fraud)
    print(lg.policy_csl_vs_fraud)
    print(lg.policy_state_vs_fraud)

    all_data = [False, False, False, False, False, False]

    clfs = [lg,
            SVC_(all_data=all_data[0]),
            KNN(all_data=all_data[1]),
            DecisionTree(all_data=all_data[2]),
            RandomForest(all_data=all_data[3]),
            NN(all_data=all_data[4])]

    experiment = False
    if experiment:
        with open("experiment.txt", 'w+') as file:
        # file = open("experiment.txt", 'w+')
            for clf in clfs:
                clf.fit()

                file.write(str(clf))
                file.write("Cross Validation Score(5 fold 20%): {}".format(clf.cross_valid()))

    # file.close()

    # fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    # score curves, each time with 20% data randomly selected as a validation set.
    # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    performance = False

    if performance:
        for clf in clfs:
            if not experiment:
                clf.fit()
            if clf.classifier_name == "Neural Network MLP":
                plt = plot_learning_curve(clf._clf, "Learning Curves " + clf.classifier_name, pd.DataFrame(clf.X).fillna(0).to_numpy(), pd.DataFrame(clf.y).fillna(0).to_numpy())
                plt.savefig("plots/performance/" + clf.classifier_name+ clf.par)


    boost_bagg = False

    if boost_bagg:
        with open("Boosting_Bagging_experiment.txt", "w+") as file:
            for clf in clfs:
                if clf.classifier_name != "SVM" and clf.classifier_name != "K-NN" and clf.classifier_name != "Neural Network MLP":
                    boost = Ada(clf)
                    bag = Bag(clf)

                    # boost_bag = Bag(Ada(clf))

                    boost.fit()
                    bag.fit()
                    # boost_bag.fit()

                    file.write(str(boost))
                    file.write("Cross Validation Score(5 fold 20%): {}".format(boost.cross_valid()))

                    plt = plot_learning_curve(boost._clf, "Learning Curves Boost " + boost.base_clf.classifier_name,
                                              pd.DataFrame(boost.X).fillna(0).to_numpy(),
                                              pd.DataFrame(boost.y).fillna(0).to_numpy())
                    plt.savefig("plots/performance/boost_" + boost.base_clf.classifier_name + boost.base_clf.par)

                    file.write(str(bag))
                    file.write("Cross Validation Score(5 fold 20%): {}".format(bag.cross_valid()))


                    plt = plot_learning_curve(bag._clf, "Learning Curves Bag " + bag.base_clf.classifier_name,
                                              pd.DataFrame(bag.X).fillna(0).to_numpy(),
                                              pd.DataFrame(bag.y).fillna(0).to_numpy())
                    plt.savefig("plots/performance/bag_" + bag.base_clf.classifier_name + bag.base_clf.par)

                    # file.write(str(boost_bag))
                    # file.write("Cross Validation Score(5 fold 20%): {}".format(boost_bag.cross_valid()))
                    #
                    #
                    # plt = plot_learning_curve(boost_bag._clf, "Learning Curves  Boost_Bag " + boost_bag.base_clf.base_clf.classifier_name,
                    #                           pd.DataFrame(boost_bag.X).fillna(0).to_numpy(),
                    #                           pd.DataFrame(boost_bag.y).fillna(0).to_numpy())
                    # plt.savefig("plots/performance/boost_bag_" + boost_bag.base_clf.base_clf.classifier_name + boost_bag.base_clf.base_clf.par)


    test_leaves = False
    if test_leaves:
        leaves = [i for i in range(1, 51)]
        clf = DecisionTree(all_data=all_data[3])
        test = clf.test_leaves()
        plt.plot([i for i in range(len(test))], test)
        plt.xlabel("Leaf Size")
        plt.ylabel("cross_validation")
        plt.title("optimal leaf size")
        plt.savefig("plots/Decision_tree_leaf_size")


    optimal_random = False

    if optimal_random:
        clf = RandomForest(all_data=all_data[4])
        clf.forest_optim_graph()

    k_nn_test = False

    if k_nn_test:
        clf = KNN(all_data=all_data[1])

        clf.test_neigboursize_manhattan()

    nn_test = False

    if nn_test:
        clf =  NN(all_data=all_data[4])

        clf.test_hidden_layers()