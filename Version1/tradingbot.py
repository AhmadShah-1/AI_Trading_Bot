# Broker
from lumibot.brokers import Alpaca

# Framework for backtesting
from lumibot.backtesting import YahooDataBacktesting

# Actual Trading bot
from lumibot.strategies.strategy import Strategy

# Deployment capabilities to go live
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api.rest import REST
from timedelta import Timedelta
from finbert_utils import estimate_sentiment

API_KEY = "PK4AFCPLTY9D4A2S5XCO"
API_SECRET = "L1g2tiTHerwSdOJILl8XoG8aXaZZ0LRrOwUM2HsY"
BASE_URL = "https://paper-api.alpaca.markets/v2"

# Paper trading means not using real money
ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True,
    "BASE_URL": BASE_URL
}


class MLTrader(Strategy):
    # Runs on start
    # Cash at risk is how much cash you are willing to risk for a buy
    def initialize(self, symbol:str="SPY", cash_at_risk:float=0.5):
        self.symbol = symbol
        self.sleeptime = "24H"    # How often to trade
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price,0)
        return cash, last_price, quantity

    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=5)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')


    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol,
                                 start=three_days_prior,
                                 end=today)

        news = [ev.__dict__["_raw"]["headline"] for ev in news]

        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment


    # Runs on each trading iteration
    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

        if cash > last_price:
            if sentiment == "positive" and probability > 0.999:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.20,
                    stop_loss_price=last_price * 0.95
                )
                self.submit_order(order)
                self.last_trade = "buy"

            elif sentiment == "negative" and probability > 0.999:
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * 0.8,
                    stop_loss_price=last_price * 1.05
                )
                self.submit_order(order)
                self.last_trade = "sell"



# Time period for backtesting
start_date = datetime(2020, 1, 3)
end_date = datetime(2023, 12, 31)



broker = Alpaca(ALPACA_CREDS)
strategy = MLTrader(name='mlstrat', broker=broker,
                    parameters={"symbol": "SPY", "cash_at_risk": 0.5})


# Test the strategy
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": "SPY", "cash_at_risk": 0.5}
)

'''
trader = Trader()
trader.add_strategy(strategy)
trader.runall()

'''