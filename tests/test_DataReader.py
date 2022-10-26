import unittest
from mlforall.DataReader import ReadData

class TestDataReader(unittest.TestCase):

    dir_in = 'C:/Users/E706657/OneDrive - Mondragon Unibertsitatea/Mondragon/4/Progra/MLForAll'
    
    route_csv = dir_in + '/test_data/Maths.csv'
    route_excel = dir_in + '/test_data/IPC.xlsx'
    route_npy = dir_in + '/test_data/Maths.npy'

    data_reader_csv = ReadData(route=route_csv)
    data_reader_excel = ReadData(route=route_excel)
    data_reader_npy = ReadData(route=route_npy)

    def test_get_file_extension(self):
        
        extension1 = 'csv'
        extension2 = 'xlsx'
        extension3 = 'npy'
        good_extensions = [extension1, extension2, extension3]

        check_extension1 = self.data_reader_csv._get_file_extension()
        check_extension2 = self.data_reader_excel._get_file_extension()
        check_extension3 = self.data_reader_npy._get_file_extension()

        check_extensions = [check_extension1, check_extension2, check_extension3]
        
        self.assertEqual(good_extensions, check_extensions)
    

    def test_read_data(self):
        
        nice_first_csv_line = ['GP','F',18,'U','GT3','A',4,4,'at_home','teacher','course','mother',2,2,0,'yes','no','no','no','yes','yes','no','no',4,3,4,1,1,3,6,5,6,6]
        nice_last_csv_line = ['MS','M',19,'U','LE3','T',1,1,'other','at_home','course','father',1,1,0,'no','no','no','no','yes','yes','yes','no',3,2,3,3,3,5,5,8,9,9]

        nice_first_npy_line = [18, 4, 4, 2, 2, 0, 4, 3, 4, 1, 1, 3, 6, 5, 6, 6]
        nice_last_npy_line = [19, 1, 1, 1, 1, 0, 3, 2, 3, 3, 3, 5, 5, 8, 9, 9]

        good_lines = [nice_first_csv_line, nice_first_npy_line, nice_last_csv_line, nice_last_npy_line]


        self.data_reader_csv.read_data()
        # check_excel = self.data_reader_excel.read_data()
        self.data_reader_npy.read_data()

        check_first_csv_line = self.data_reader_csv.data.iloc[0, :].tolist()
        check_last_csv_line = self.data_reader_csv.data.iloc[-1, :].tolist()

        check_first_npy_line = self.data_reader_npy.data.iloc[0, :].tolist()
        check_last_npy_line = self.data_reader_npy.data.iloc[-1, :].tolist()

        check_lines = [check_first_csv_line, check_first_npy_line, check_last_csv_line, check_last_npy_line]


        self.assertEqual(good_lines, check_lines)
