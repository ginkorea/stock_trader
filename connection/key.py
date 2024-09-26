from const.auth import alpaca_sk, alpaca_pk


class Key:
    def __init__(self, sk, pk):
        self.__pk = pk  # Private attribute
        self.__sk = sk  # Private attribute

    def get_key(self):
        return self.__pk  # Access private attribute

    def get_secret(self):
        return self.__sk  # Access private attribute


alpaca_key = Key(alpaca_sk, alpaca_pk)
