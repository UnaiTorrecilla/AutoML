import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

# from xgboost import XGBClassifier, XGBRegressor
# from lightgbm import LGBMClassifier, LGBMRegressor
# from catboost import CatBoostRegressor, CatBoostClassifier


from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error


class ModelData:
    

    def __init__(self, train_data: pd.DataFrame, target_data: pd.DataFrame, test_size: float=0.2):

        self.train_data = train_data
        self.target_data = target_data
        self.test_size = test_size

        self.xtr, self.xte, self.ytr, self.yte = train_test_split(self.train_data.values, self.target_data.values, 
        test_size=self.test_size, stratify=None if self.is_target_variable_numeric() else self.target_data)


    def is_target_variable_numeric(self):
        return is_numeric_dtype(self.target_data)


    @staticmethod
    def _create_regression_models_pool():

        return {
            # 'xgb': XGBRegressor(),
            'rf': RandomForestRegressor()
            # 'lgbm': LGBMRegressor(),
            # 'catboost': CatBoostRegressor()
        }
    

    @staticmethod
    def _create_classification_models_pool():
        
        return  {
            # 'xgb': XGBClassifier(),
            'rf': RandomForestClassifier()
            # 'lgbm': LGBMRegressor(),
            # 'catboost': CatBoostClassifier()
        }


    def create_models_pool(self):
        '''
        Function to create a pool of models.
        '''
        if self.is_target_variable_numeric():
            return self._create_regression_models_pool()
        else:
            return self._create_classification_models_pool()
            

    def train_algorithm_and_return_predictions(self, model, xtr_scaled: np.ndarray=None, xte_scaled: np.ndarray=None):
        '''
        Function to fit a model and return the predictions.
        Parameters
        ----------
        model: sklearn model
            Model to fit.
        xtr_scaled: numpy.ndarray
            Scaled training data. If None, the unscaled training data will be used.
        xte_scaled: numpy.ndarray
            Scaled test data. If None, the unscaled test data will be used.
        '''
        if xtr_scaled is None:
            xtr_scaled = self.xtr
        if xte_scaled is None:
            xte_scaled = self.xte

        model.fit(xtr_scaled, self.ytr)
        return model.predict(xte_scaled)
    

    def _evaluate_regression_model(self, model, predictions: np.ndarray):
        '''
        Function to evaluate the performance of a regression model.
        Parameters
        ----------
        model: sklearn model
            Model to evaluate.
        predictions: numpy.ndarray
            Predictions of the model.
        
        Returns
        -------
        train_score: float
            Score of the model on the training data.
        test_r2_score: float
            R2 score of the model on the test data.
        test_mse_score: float
            Mean squared error of the model on the test data.
        test_mae_score: float
            Mean absolute error of the model on the test data.
        '''
        cross_score = cross_val_score(model, self.xtr, self.ytr, cv=5, scoring='r2')
        train_score = np.mean(cross_score)
        test_r2_score = r2_score(self.yte, predictions)
        test_mse_score = mean_squared_error(self.yte, predictions)
        test_mae_score = mean_absolute_error(self.yte, predictions)

        return {'train_r2_5cv': train_score, 'test_r2': test_r2_score, 
                'test_mse': test_mse_score, 'test_mae': test_mae_score}
    

    def _evaluate_classification_model(self, model, predictions: np.ndarray):
        '''
        Function to evaluate the performance of a classification model.
        Parameters
        ----------
        model: sklearn model
            Model to evaluate.
        predictions: numpy.ndarray
            Predictions of the model.
        
        Returns
        -------
        train_score: float
            Score of the model on the training data.
        test_accuracy_score: float
            Accuracy score of the model on the test data.
        test_roc_auc_score: float
            ROC AUC score of the model on the test data.
        '''

        cross_score = cross_val_score(model, self.xtr, self.ytr, cv=5, scoring='roc_auc')
        train_score = np.mean(cross_score)
        test_accuracy_score = accuracy_score(self.yte, predictions)
        test_roc_auc_score = roc_auc_score(self.yte, predictions)

        return {'train_auc': train_score, 'test_acc': test_accuracy_score, 'test_auc': test_roc_auc_score}


    def evaluate_model(self, model, predictions: np.ndarray):
        '''
        Function to evaluate the performance of a model.
        '''
        if self.is_target_variable_numeric():
            return self._evaluate_regression_model(model=model, predictions=predictions)
        else:
            return self._evaluate_classification_model(model=model, predictions=predictions)
