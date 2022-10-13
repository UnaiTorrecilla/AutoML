import pandas as pd
import numpy as np

class CleanPandas:


    def __init__(self, route: str) -> None:
        self.route = route
    
    
    def _get_file_extension(self) -> str:
        extension = self.route.split('.')[-1]
        print(f'Identified .{extension} extension. If this is not correct, please review that '
              f'the extension is held after the last "." of the inserted route.')
        return extension


    def _read_csv(self, **kwargs):
        # self.data = pd.read_csv()
        return None


class CleanNumpy:
    pass


class CleanText:
    pass 
