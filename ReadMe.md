## How to run the code


> Run file src/fetch_&_handle_data.py, swap parameters as required in the produce_indicators(ticker, periods, interval) func.

- Valid **tickers**: ticker names for stocks, crypos, ETFs, etc
- Valid **periods**: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
- Valid **intervals**: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

<br>

yfinance doc: [https://pypi.org/project/yfinance/](https://pypi.org/project/yfinance/)

plotly doc: [https://plotly.com/python/](https://plotly.com/python/)

<br>

####Example using parameters: ('BTC-USD', '5y', '1d'):
![Example](https://raw.githubusercontent.com/CaffeinatedKev/stock-crypto-indicators/master/example.png)
