import pandas as pd


class TickerList:

    def __init__(self, file='data/nasdaq_screener_1702740518530.csv'):
        self.df = pd.read_csv(file)
        self.tickers = self.df['Symbol']
        self.names = self.df['Name']
        self.dict = dict(zip(self.tickers, self.names))


tickers_df = pd.read_csv("tickers.csv", header=None)

# Convert the single row of tickers into a list and remove spaces
tickers = [ticker.strip() for ticker in tickers_df.iloc[0, :]]

tickers_df.to_csv('tickers.csv', index=False, header=False)
