import yfinance as yf


SYMBOL = "2330.TW"
HISTORY = "10d"
# df=yf.download(SYMBOL, [HISTORY, "1d"])  
df = yf.download('TSLA',period=HISTORY, interval="1d")
print(df)
# all_day_k = yf.Ticker(SYMBOL).history(period=HISTORY, interval="1d")