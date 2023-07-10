import pandas as pd


class CSVToDataframe:
    def __init__(self, filename):
        self.filename = filename
        if filename[-4:] != '.csv':
            filename = filename + '.csv'
        self.dataframe = pd.read_csv(filename)

    def getDataframe(self):
        return self.dataframe