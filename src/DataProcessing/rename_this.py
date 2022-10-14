import pandas as pd
import numpy as np
import sys

class CleanPandas:


    def __init__(self, route: str) -> None:
        self.route = route
    
    
    def _get_file_extension(self) -> str:
        '''
        Returns the file extension of the file
        '''
        extension = self.route.split('.')[-1]
        print(f'Identified .{extension} extension. If this is not correct, please review that '
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
                'Please input the separator used in the csv file, or continue if it is already correct: ')
                
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

        

    def _read_excel(self, **kwargs):
        self.data = pd.read_excel(self.route, **kwargs)

        # If the data has only one column, it is probably due to the separator being wrong.
        self._check_data_dimensions(**kwargs)
    

    def _read_json(self, **kwargs):
        self.data = pd.read_json(self.route, **kwargs)

        # Check that the json format is appropiate for Pandas to read it.
        


        # If the data has only one column, it is probably due to the separator being wrong.
        self._check_data_dimensions(**kwargs)


    def _read_parquet(self, **kwargs):
        self.data = pd.read_parquet(self.route, **kwargs)

        # If the data has only one column, it is probably due to the separator being wrong.
        self._check_data_dimensions(**kwargs)


    def read_data(self):
        extension = self._get_file_extension()
        if extension == 'csv':
            self._read_csv()
        elif extension == 'xlsx':
            self._read_excel()
        elif extension == 'json':
            self._read_json()
        elif extension == 'parquet':
            self._read_parquet()
        else:
            raise NotImplementedError(f'Extension {extension} not implemented. Please review the '
                                      f'available extensions in the documentation.')


class CleanNumpy:
    pass


class CleanText:
    pass 


if __name__ == '__main__':
    hi = CleanPandas(sys.argv[1])
    # hi._get_file_extension()
    hi._read_csv()
    print(hi.data.head())

