from typing import Union
import pandas as pd
import numpy as np
import warnings
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer, Normalizer



class ScaleData:
    '''
    This class is used to scale the data. It is used in the DataModeler class.
    '''

    def __init__(self, data: pd.DataFrame, target: Union[str, int], scaling_method, scaling_params) -> None:
        '''
        Parameters
        ----------
        data: pandas.DataFrame
            Data to scale.
        target: str or int
            Name or index of the target column.
        '''

        self.data = data
        self.target = target
        self._separate_target_and_training_data()


    def _separate_target_and_training_data(self) -> pd.DataFrame:
        
        if isinstance(self.target, str):
            self.target = self.data.columns.get_loc(self.target)
        
        self.target_data = self.data.iloc[:, self.target]
        self.train_data = self.data.drop(self.data.columns[self.target], axis=1)

        return None
    

    def _select_dtypes(self) -> pd.DataFrame:
        
        previous_num_columns = self.train_data.shape[1]
        self.train_data = self.train_data.select_dtypes(include=np.number)

        new_num_columns = self.train_data.shape[1]
        if new_num_columns != previous_num_columns:
            warnings.warn(f'{previous_num_columns - new_num_columns} columns were removed due to not being numeric.'
                          ' In a future release the categorical variables will also be treated.')

        return self.train_data


    def _clean_data(self) -> pd.DataFrame:
        imputer = SimpleImputer(strategy='median')

        warnings.warn('The data will be cleaned by replacing the missing values with the median of the column. In future releases more'
                      ' advanced cleaning methods will be implemented.')

        if self.train_data.shape[0] == 1:
            data_array = self.train_data.values.reshape(1, -1)
        elif self.train_data.shape[1] == 1:
            data_array = self.train_data.values.reshape(-1, 1)
        else:
            data_array = self.train_data.values

        return pd.DataFrame(imputer.fit_transform(data_array), columns=self.train_data.columns)


    @staticmethod
    def create_scaling_methods_pool():
        '''
        This method creates a dictionary with the scaling methods available in the class.
        '''

        return {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'maxabs': MaxAbsScaler(),
            'quantile': QuantileTransformer(),
            'power': PowerTransformer(),
            'normalizer': Normalizer()
        }

    
    def _scale_data(self, scaler: Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, 
                                    QuantileTransformer, PowerTransformer, Normalizer]) -> pd.DataFrame:
        '''
        This method scales the data using the scaling method selected by the user.
        '''

        return scaler.fit_transform(self.train_data)
    
    def main(self, scaler: Union[StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
                                QuantileTransformer, PowerTransformer, Normalizer]) -> pd.DataFrame:

        self._select_dtypes()
        self._clean_data()

        scaled_data = self._scale_data(scaler)
        
        return scaled_data
