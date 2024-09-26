from connection.client import APIConnection

class Agent:

    def __init__(self, name=None, portfolio=None, is_paper=True, default_key=True):
        if name is None: self.name = "Alice"
        else: self.name = name
        self.portfolio = portfolio
        self.clients = APIConnection(is_paper=is_paper, default_key=default_key)
        self.all_tickers = self.clients.trading_client.get_all_assets()
        self.account = self.clients.trading_client.get_account()

agent_alice = Agent()
print(f"Agent {agent_alice.name} has account number {agent_alice.account}")
