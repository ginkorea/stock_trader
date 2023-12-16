import pandas as pd


class TickerList:

    def __init__(self, file='data/nasdaq_screener_1702740518530.csv'):
        self.df = pd.read_csv(file)
        self.tickers = self.df['Symbol']
        self.names = self.df['Name']
        self.dict = zip(self.tickers, self.names)

