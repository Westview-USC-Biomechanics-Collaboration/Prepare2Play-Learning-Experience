import numpy
import pandas

class FileFormatter:
    def __init__(self):
        """
        names = ["abs time (s)", "Fx1", "Fy1", "Fz1", "|Ft1|", "Ax1", "Ay1", "COM px1", "COM py1", "COM pz1",
                 "Fx2", "Fy2", "Fz2", "|Ft2|", "Ax2", "Ay2", "COM px2", "COM py2", "COM pz2"]
        """
        self.columns = ["abs time (s)", "Fx1", "Fy1", "Fz1", "|Ft1|", "Ax1", "Ay1", "COM px1", "COM py1", "COM pz1",
                        "Fx2", "Fy2", "Fz2", "|Ft2|", "Ax2", "Ay2", "COM px2", "COM py2", "COM pz2"]

    def __columnchecker(self, df: pandas.DataFrame):
        """
        Check if the columns of the dataframe are correct
        :param df: pandas dataframe
        :return: True if the columns are correct, False otherwise
        """
        if len(df.columns) != len(self.columns):
            print("The number of columns is not correct")
            return False
        for i in range(len(self.columns)):
            if df.columns[i] != self.columns[i]:
                print(f"The column {i} is not correct")
                return False
        return True
    def readTxt(self, filePath: str):
        """
        Read a txt file and return a pandas dataframe
        :param filePath: path to the txt file
        :return: pandas dataframe
        """
        data = numpy.loadtxt(filePath, skiprows=19)
        df = pandas.DataFrame(data, columns=self.columns)
        if not self.__columnchecker(df):
            print("The columns are not correct")
            raise ValueError("The columns are not correct")
        return df
    
    def readCsv(self, filePath: str):
        """
        Read a csv file and return a pandas dataframe
        :param filePath: path to the csv file
        :return: pandas dataframe
        """
        df = pandas.read_csv(filePath, skiprows=19)
        if not self.__columnchecker(df):
            print("The columns are not correct")
            raise ValueError("The columns are not correct")
        return df
    
    def readExcel(self, filePath: str):
        """
        Read an excel file and return a pandas dataframe
        :param filePath: path to the excel file
        :return: pandas dataframe
        """
        df = pandas.read_excel(filePath, skiprows=18)
        if len(df.columns) != len(self.columns):
            print("The number of columns is not correct")
            raise ValueError("The number of columns is not correct")
        df.columns = self.columns

        return df
    


if __name__ == "__main__":
    converter = FileFormatter()
    df = converter.readExcel(r"C:\Users\chase\Downloads\ajp_lr_JN_for05_Raw_Data.xlsx")
    print(df.head())
    for i in range(50):
        print(df.iloc[i])