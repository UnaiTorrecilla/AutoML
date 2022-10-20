from typing import Union
import warnings
import pandas as pd
import numpy as np
import sys
from sklearn.impute import SimpleImputer


def select_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    previous_num_columns = data.shape[1]
    data = data.select_dtypes(include=np.number)

    new_num_columns = data.shape[1]
    if new_num_columns != previous_num_columns:
        warnings.warn(f'{previous_num_columns - new_num_columns} columns were removed due to not being numeric.'
                      ' In a future release the categorical variables will also be treated.')

    return data

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    imputer = SimpleImputer(strategy='median')

    if data.shape[0] == 1:
        data_array = data.values.reshape(1, -1)
    elif data.shape[1] == 1:
        data_array = data.values.reshape(-1, 1)
    else:
        data_array = data.values

    return pd.DataFrame(imputer.fit_transform(data_array), columns=data.columns)

class ReadData:


    def __init__(self, route: str) -> None:
        self.route = route
    
    
    def _get_file_extension(self) -> str:
        '''
        Returns the file extension of the file
        '''
        extension = self.route.split('.')[-1]
        print(f'\nIdentified .{extension} extension. If this is not correct, please review that '
              f'the extension is held after the last "." of the inserted route.')
        return extension
    


    def _proceed_recursivity_if_needed(self, sep: str, **kwargs) -> None:
        if sep == 'continue':
                print(f'Continuing with the inferred separator ({kwargs.get("sep", ",")})...')
        else:
            self._read_csv(sep=sep, **kwargs)


    def _check_data_dimensions(self, **kwargs):
        if self.data.shape[1] == 1:
            print(f'\nFound {self.data.shape[1]} column on the data. This is probably due to the separator being wrong.'
                ' If this is correct please input "continue" when asked, if not, type the correct separator.')

            # We delete the previous separator if it was given to avoid errors
            kwargs.pop('sep', None)
            # of duplicated arguments due to the recursive call.

            sep = input(
                'Please input the separator used in the csv file, or "continue" if it is already correct: ')
                
            self._proceed_recursivity_if_needed(sep, **kwargs)

    def _read_csv(self, **kwargs) -> None:
        '''
        This method reads a csv file and stores it in the attribute self.data.
        It tries to infer the separator, but if it fails, it will ask the user to input it.
        The function accepts all the parameters of the pandas.read_csv function.
        
        Parameters
        ----------
        **kwargs: dict
            Parameters of the pandas.read_csv function if desired. If not, the default parameters will be used.
        
        Returns
        -------
        None
        '''
        
        self.data = pd.read_csv(self.route, **kwargs)

        # If the data has only one column, it is probably due to the separator being wrong.
        self._check_data_dimensions(**kwargs)

    

    def _read_excel(self, sheet_name: Union[int, str]=0,**kwargs):
        '''
        This method reads an excel file and stores it in the attribute self.data.
        
        Parameters
        ----------
        sheet_name: int or str.
            The sheet name or number to read from the excel file. If not specified, the first sheet will be read.
            It is not yet supported to read multiple sheets at once.

        **kwargs: dict
            Parameters of the pandas.read_excel function if desired. If not, the default parameters will be used.
        '''
        if not isinstance(sheet_name, int) and not isinstance(sheet_name, str):
            raise NotImplementedError(f'The sheet_name parameter must be an integer or a string. '
                                        f'Found {type(sheet_name)}. Please review the documentation.')
        self.data = pd.read_excel(self.route, sheet_name=sheet_name, **kwargs)

    

    # def _read_json(self, **kwargs):
    #     '''
    #     This method reads a json file and stores it in the attribute self.data.

    #     Parameters
    #     ----------
    #     **kwargs: dict
    #         Parameters of the pandas.read_json function if desired. If not, the default parameters will be used.
    #     '''
    #     try:
    #         self.data = pd.read_json(self.route, **kwargs)
    #     except ValueError:
    #         self.data = pd.read_json(self.route, lines=True, **kwargs)

    def _read_parquet(self, **kwargs):
        '''
        This method reads a parquet file and stores it in the attribute self.data.

        Parameters
        ----------
        **kwargs: dict
            Parameters of the pandas.read_parquet function if desired. If not, the default parameters will be used.
        '''
        self.data = pd.read_parquet(self.route, **kwargs)
    
    def _read_numpy(self, **kwargs):
        '''
        This method reads a numpy file and stores it in the attribute self.data. It only accepts .npy
        file extensions.

        Parameters
        ----------
        **kwargs: dict
            Parameters of the np.load function if desired. If not, the default parameters will be used.
        '''
        self.data = pd.DataFrame(np.load(self.route, **kwargs))
        


    def read_data(self):
        extension = self._get_file_extension()
        if extension == 'csv' or extension == 'txt':
            self._read_csv()
        elif extension == 'xlsx' or extension == 'xls':
            self._read_excel()
        # elif extension == 'json':
        #     self._read_json()
        elif extension == 'parquet':
            self._read_parquet()
        elif extension == 'npy':
            self._read_numpy()
        else:
            raise NotImplementedError(f'Extension {extension} not implemented. Please review the '
                                      f'available extensions in the documentation.')
        
        print(self.data.head())

        print('\nThe previous is a sample of the found data. If it is not correct please run the script again'
                ' and select the proper options.')
        
        return self.data


    


if __name__ == '__main__':
    hi = ReadData(sys.argv[1])
    data = hi.read_data()
    data_cleaned = clean_data(select_dtypes(data))

