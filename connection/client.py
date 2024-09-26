from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
import connection.key as ky


class APIConnection:

    def __init__(self, is_paper=True, default_key=True):
        """
        Initialize API connection to both Alpaca Trading and Historical Data APIs.
        Exposes both the trading and historical data clients for direct use.

        :param is_paper: Use paper trading environment if True (default: True)
        :param default_key: Use default keys from the `key` module if True (default: True)
        """
        self.endpoint = "https://paper-api.alpaca.markets" if is_paper else "https://api.alpaca.markets"

        # Use default key from the key module or prompt for user input
        if default_key:
            self.auth_key = ky.alpaca_key
        else:
            sk, pk = self.get_key_from_user()
            self.auth_key = ky.Key(sk, pk)

        # Initialize Historical Data Client
        self.historical_data_client = StockHistoricalDataClient(
            api_key=self.auth_key.get_key(),
            secret_key=self.auth_key.get_secret()
        )

        # Initialize Trading Client
        self.trading_client = TradingClient(
            api_key=self.auth_key.get_key(),
            secret_key=self.auth_key.get_secret(),
            paper=is_paper  # This determines if it's live or paper trading
        )

    @staticmethod
    def get_key_from_user():
        """Prompt user to input Alpaca API keys manually."""
        pk = input("Please enter your public key:")
        sk = input("Please enter your secret key:")
        return sk, pk
