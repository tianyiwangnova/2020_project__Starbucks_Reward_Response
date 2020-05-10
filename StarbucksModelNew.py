import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


class StarbucksModel:

    def __init__(self, data_cleaning_pipeline):
        self.data_cleaning_pipeline = data_cleaning_pipeline
        self.data_cleaning_pipeline_fit = False
        self.data_scaled = False


    @staticmethod
    def customize_train_test_split(data,
                                   tag_col,
                                   minority_tag_value):
        
        """
        Do train_test_split on minority group and majority group seperately
        -- sample 20% minority and 20% majority to the final test set
        
        This process doesn't drop any columns
        
        Input:
            data: the data to split
            tag_col: column for the class label
            minority_tag_value: class label that has very few representatives
        """
        
        data_minor = data[data[tag_col] == minority_tag_value]
        data_major = data[data[tag_col] != minority_tag_value]
        
        train_minor, test_minor = train_test_split(data_minor, test_size=0.2)
        train_major, test_major = train_test_split(data_major, test_size=0.2)
        
        train = pd.concat([train_minor, train_major]).sample(frac=1)
        test = pd.concat([test_minor, test_major]).sample(frac=1)
        
        return train, test


    def train_model(self, X, y, estimator, param_grid):
    
        """
        Tune and train the estimator and choose the best parameters from candidate parameter sets;
        It will return a GridSearchCV object
        
        Input:
        estimator: a sklearn estimator
        params_prid: Dictionary with parameters names (string) as keys and lists of parameter 
                     settings to try as values, or a list of such dictionaries, in which case 
                     the grids spanned by each dictionary in the list are explored. This enables 
                     searching over any sequence of parameter settings
        """

        X = self.data_cleaning_pipeline.fit_transform(X)
        self.data_cleaning_pipeline_fit = True

        self.model = GridSearchCV(estimator=estimator,
                                  param_grid=param_grid,
                                  scoring='precision',
                                  verbose=1)
        
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        self.data_scaled = True
        X = self.scaler.transform(X)
        
        self.model.fit(X, y)

        return self.model


    def test_model(self, X, y):
    
        """
        A general function for checking the f1, precision and recall score of the model;
        
        Output is a dictionary with 'precision', 'recall' and 'f1'
        """
        X = self.data_cleaning_pipeline.transform(X)
        X = self.scaler.transform(X)
        y_pred = self.model.predict(X)
        result = {}
        result['precision'] = precision_score(y, y_pred)
        result['recall'] = recall_score(y, y_pred)
        result['f1'] = f1_score(y, y_pred)
        return result


    def feature_importance(self, estimator, X, y):

        '''
        Train a model and get the feature importance matrix
        '''

        columns = X.columns

        if self.data_cleaning_pipeline_fit:
            X = self.data_cleaning_pipeline.transform(X)
        else:
            X = self.data_cleaning_pipeline.fit_transform(X)

        if self.data_scaled:
            X = self.data_cleaning_pipeline.transform(X)
        else:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        X = self.scaler.transform(X)
        model = estimator.fit(X, y)
        feature_importance = pd.DataFrame({'features': columns, 
                                           'importance': model.feature_importances_})\
                             .sort_values('importance', ascending=False)
        return feature_importance


    def predict(self,
                test_data):
        """
        Make prediction on test_data;
        The input test_data should have the same columns as the X for train_model
        """
        test_data = self.data_cleaning_pipeline.transform(test_data)
        test_data = self.scaler.transform(test_data)
        y = self.model.predict_proba(test_data)[:,1]
        return y


    def train_full(self, offer_pipe, estimator, params):

        """
        Do train_test_split on the input data, tune parameters, show the performance of the model
        on the testing set, show the important features and plot violin plots to show how the 
        important features vary among the positive and negative samples.
        """

        train, test = self.customize_train_test_split(offer_pipe, 
                                                      'completed', 
                                                      True)
        self.train_model(train.drop('completed', axis=1), train['completed'], estimator, params)
        print("===Model===")
        print(self.model.best_estimator_)

        print("===Result on Testing set===")
        print(self.test_model(test.drop('completed', axis=1), test['completed']))

        feature_importance = self.feature_importance(estimator, 
                                                     train.drop('completed', axis=1), 
                                                     train['completed'])
        print("===Important features===")
        print(feature_importance[:5])

        most_important_6 = list(feature_importance['features'][:6])

        plt.figure(figsize=(20,10))
        plt.subplot(231)
        sns.violinplot(x="completed", y=most_important_6[0], data=train)
        plt.title(most_important_6[0])
        plt.subplot(232)
        sns.violinplot(x="completed", y=most_important_6[1], data=train)
        plt.title(most_important_6[1])
        plt.subplot(233)
        sns.violinplot(x="completed", y=most_important_6[2], data=train)
        plt.title(most_important_6[2])
        plt.subplot(234)
        sns.violinplot(x="completed", y=most_important_6[3], data=train)
        plt.title(most_important_6[3])
        plt.subplot(235)
        sns.violinplot(x="completed", y=most_important_6[4], data=train)
        plt.title(most_important_6[4])
        plt.subplot(236)
        sns.violinplot(x="completed", y=most_important_6[5], data=train)
        plt.title(most_important_6[5])

        return feature_importance




    