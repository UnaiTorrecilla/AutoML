.. mlforall documentation master file, created by
   sphinx-quickstart on Wed Oct 26 16:07:07 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mlforall's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Description
-----------
**mlforall** is an open-source library aimed to developers that are beginners in the data analysis area but want to build powerful machine learning projects from the very beginning. The package offers a reliable, easy to use and well documented set of functions that drive the user through the most common steps of any machine learning projects, from data reading to model testing.

Main features
-------------
These are some of the functionalities that mlforall offers:  


1. File extension asbtraction when reading data (only supported for `.csv`, `.txt`, `.xls`, `.xlsx`, `.parquet` and `.npy`)
2. Automatic handling of non-numeric features and missing values.
3. A pool with almost all the data-scaling methods available and the most common ML models.
4. Automatic model evaluation and reporting.


Classes and functions
=====================
The module is divided in three main classes that are DataReader, DataScaler and DataModeler. Their methods and parameters are as follows:

DataReader
----------
This class is used to read data from a file with very high-level execution code.

.. py:function:: def __init__(route):

   It is the constructor of the class.

   :param route: *Mandatory*. The route of the file to be read.
   :type route: str

.. py:function:: def read_data():
   
   It reads the data from the file and returns a pandas DataFrame. It has file extension asbtraction \
   for the supported file types (`.csv`, `.txt`, `.xls`, `.xlsx`, `.parquet` and `.npy`).

   :return: The data read from the file.
   :rtype: pandas.DataFrame  


DataScaler
----------
This class implements all the necessary methods to scale the data. It has a pool with the most \
common data-scaling methods available.

.. py:function:: def __init__(data, target):

   It is the constructor of the class. It takes the data to be scaled and the target column. \
   When this is executed, the class internally separates target and training data into \
   different variables to ensure that the target column is not scaled.

   :param data: *Mandatory*. The data to be scaled.
   :type data: pandas.DataFrame
   :param target: *Mandatory*. The name of the target column.
   :type target: str

.. py:function:: def select_dtypes():

   It automatically selects a subset of the original training data that only contains the numeric columns. \
   If some columns are dropped, a warning is shown.

   :return: The subset of the original training data that only contains the numeric columns.
   :rtype: pandas.DataFrame

.. py:function:: def clean_data():

   It automatically imports missing values. The current strategy is to replace them with the median \
   of the column.

   :return: The training data with the missing values replaced.
   :rtype: pandas.DataFrame
   

.. py:function:: def create_scaling_methods_pool():

   It creates a pool with the most common data-scaling methods available. The current pool is: \
   `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, `QuantileTransformer`, \
   `PowerTransformer` and `Binarizer`.

   :return: The pool with the data-scaling methods.
   :rtype: dict


.. py:function:: def scale_data(scaler, xtr, xte):

   It scales the data using the selected scaler. It trains the scaler with the training data and \
   then applies the same transformation to the test data. The current supported scalers are: \
   `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`, \
   `QuantileTransformer` and `PowerTransformer`.

   :param scaler: *Mandatory*. The scaler to be used.
   :type scaler: One of the mentioned sklearn scalers
   :param xtr: *Mandatory*. The training data.
   :type xtr: pandas.DataFrame
   :param xte: *Mandatory*. The test data.
   :type xte: pandas.DataFrame
   :return: The scaled data.
   :rtype: pandas.DataFrame


DataModeler
-----------
This class implements all the necessary methods to train and test a model. It has a pool with the most \
common ML models available and a method to evaluate the model with the most common metrics.

.. py:function:: def __init__(train_data, target_data, test_size):

   It is the constructor of the class. It takes the training data, the target data and the test size. \
   When this is executed, the class internally separates the training data into training and test \
   data to ensure that the model is tested with unseen data. The class also takes into account if the \
   target variable is numeric or not so that it applies stratify on the target variable accordingly.

   :param train_data: *Mandatory*. The training data.
   :type train_data: pandas.DataFrame
   :param target_data: *Mandatory*. The target data.
   :type target_data: pandas.DataFrame
   :param test_size: *Optional*. The size of the test data. Default is 0.2.
   :type test_size: float

.. py:function:: def create_models_pool():

   Function to create a pool of models. It will create a pool of regression models if the target \
   variable is numeric and a pool of classification models if the target variable is categorical. \
   The options for classification models are: `LogisticRegression` and `RandomForestClassifier`, while \
   the options for regression models are: `RandomForestRegressor`.

   :return: The pool of models.
   :rtype: dict

.. py:function:: def train_agorithm_and_return_prediction(model, xtr_scaled, xte_scaled):

   Function to fit a model and return the predictions.

   :param model: *Mandatory*. The model to be used.
   :type model: One of the mentioned sklearn models
   :param xtr_scaled: *Optional*. The scaled training data. If not provided, \
   the non-scaled training data will be used.
   :type xtr_scaled: numpy.ndarray
   :param xte_scaled: *Optional*. The scaled test data. If not provided, \
   the non-scaled test data will be used.
   :type xte_scaled: numpy.ndarray
   :return: The predictions.
   :rtype: numpy.ndarray

.. py:function:: def evaluate_model(model, predictions, cv):

   Function that evaluates the performance of the model. Depending on the type of problem \
   (classification or regression) it will return some metrics or another. For regression problems \
   the calculated metrics are:

   - Train R2 score with cross-validation.
   - Test R2 score.
   - Test MSE score.
   - Test MAE score.

   In case of the classification problems the metrics are:

   - Train ROC AUC score with cross-validation.
   - Test accuracy score.
   - Test ROC AUC score.

   :param model: the trained model to be tested.
   :type model: One of the mentioned sklearn models
   :param cv: number of k-fold to perform on the cross-validation.
   :type cv: int
   :return: a dictionary with the predictions.
   :rtype: dict


