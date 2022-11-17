import numpy as np

# for dataframes
import pandas as pd

# for easier visualization
import seaborn as sns

# for visualization and to display plots
from matplotlib import pyplot as plt
# %matplotlib inline

# import color maps
from matplotlib.colors import ListedColormap

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

from math import sqrt

# to split train and test set
from sklearn.model_selection import train_test_split

# to perform hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score

# Machine Learning Models
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix

from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score

class RFT_Model:
    model = None
    def __init__(self) -> None:
        df = pd.read_csv("Indian Liver Patient Dataset (ILPD).csv")

        ## if score==negative, mark 0 ;else 1 
        def partition(x):
            if x == 2:
                return 0
            return 1
        df['Dataset'] = df['Dataset'].map(partition)

        """## Distribution of categorical data"""
        df.describe(include=['object'])

        df[df['Gender'] == 'Male'][['Dataset', 'Gender']].head()
        M, F = df['Gender'].value_counts()

        ## if score==negative, mark 0 ;else 1 
        def partition(x):
            if x =='Male':
                return 0
            return 1
        df['Gender'] = df['Gender'].map(partition)

        ## Correlations
        df.corr()
        
        ## Data Cleaning
        df = df.drop_duplicates()

        ## Removing Outliers
        df.Aspartate_Aminotransferase.sort_values(ascending=False).head()
        df = df[df.Aspartate_Aminotransferase <=3000 ]
        df.Aspartate_Aminotransferase.sort_values(ascending=False).head()
        df = df[df.Aspartate_Aminotransferase <=2500 ]
        df.isnull().values.any()
        df=df.dropna(how='any')

        ### Data Preparation
        # Create separate object for target variable
        y = df.Dataset

        # Create separate object for input features
        X = df.drop('Dataset', axis=1)

        # Split X and y into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=df.Dataset)

        # Print number of observations in X_train, X_test, y_train, and y_test
        #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        ### Data standardization

        self.train_mean = X_train.mean()
        self.train_std = X_train.std()

        ## Standardize the train data set
        X_train = (X_train - self.train_mean) / self.train_std

        ## Check for mean and std dev.
        X_train.describe()

        ## Note: We use train_mean and train_std_dev to standardize test data set
        X_test = (X_test - self.train_mean) / self.train_std

        ## Check for mean and std dev. - not exactly 0 and 1
        X_test.describe()
        
        #Random Forest Tree Model
        tuned_params = {'n_estimators': [100, 200, 300, 400, 500], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        self.model = RandomizedSearchCV(RandomForestClassifier(), tuned_params, n_iter=15, scoring = 'roc_auc', n_jobs=-1)
        self.model.fit(X_train, y_train)

        #self.model.best_estimator_

        y_train_pred = self.model.predict(X_train)

        y_pred = self.model.predict(X_test)

        # Get just the prediction for the positive class (1)
        y_pred_proba = self.model.predict_proba(X_test)[:,1]

        #### Feature Importance
        ## Building the model again with the best hyperparameters
        self.model = RandomForestClassifier(n_estimators=500, min_samples_split=2, min_samples_leaf=4)
        self.model.fit(X_train, y_train)

    def predict(self, test_data):
        test_data = (test_data - self.train_mean) / self.train_std
        pred_proba = self.model.predict_proba(test_data)[:,1]
        return pred_proba[0]
