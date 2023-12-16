import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockTradesRequest
from connector.keys import Authentication


class APIConnection:

    def __init__(self, paper=True):
        self._authentication = Authentication()
        self.historical_data_client = StockHistoricalDataClient(
            api_key=self._authentication.key,
            secret_key=self._authentication.secret
        )

    def get_stock_data(self, ticker, start_date, end_date, timeframe):
        bars_request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=timeframe,
            start=start_date,
            end=end_date
        )

        # Fetch the stock bars
        bars = self.historical_data_client.get_stock_bars(bars_request)
        df = bars.df
        df.reset_index(inplace=True)
        return df

    def get_high_resolution_stock_data(self, ticker, start_date, end_date):
        print('starting high resolution stock data request')
        trades_request = StockTradesRequest(
            symbol_or_symbols=ticker,
            start=start_date,
            end=end_date,
            # You can specify additional parameters like limit if needed
        )

        # Fetch the stock trades
        trades = self.historical_data_client.get_stock_trades(trades_request)

        # Convert the trades to a DataFrame
        high_res_data = pd.DataFrame([{
            "timestamp": trade.timestamp, "price": trade.price,
            "size": trade.size, "exchange": trade.exchange,
            "conditions": trade.conditions, "tape": trade.tape
        } for trade in trades.df[ticker]])
        print('finished processing high resolution stock data request.')

        return high_res_data

    def get_all_tickers(self):
        # Fetch the list of all active assets
        active_assets = self.list_assets(status='active')

        # Filter for assets that are tradable and of type 'us_equity'
        tradable_assets = [asset for asset in active_assets if asset.tradable and asset.asset_class == 'us_equity']

        # Extract the symbol for each asset
        tickers = [asset.symbol for asset in tradable_assets]

        return tickers
