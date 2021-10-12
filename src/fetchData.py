import requests
import json
import datetime
import time
from dateutil.relativedelta import relativedelta
import math


def get_btc_data():
    timestamp = math.trunc(time.time())
    ten_years_ago = datetime.datetime.now() - relativedelta(years=5)
    past_timestamp = math.trunc(time.mktime(ten_years_ago.timetuple()))
    request = requests.get('https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from=' + str(past_timestamp) + '&to=' + str(timestamp))
    data = request.json()
    if "prices" in data:
        formatted_data = []
        for value1, value2 in data["prices"]:
            date = datetime.datetime.fromtimestamp(value1 / 1000)
            formatted_data.append({"Date": date.strftime('%Y-%m-%d'), "Close": value2})
        with open('bitcoin-data.json', 'w') as dumpfile:
            json.dump(formatted_data, dumpfile)


def get_eth_data():
    timestamp = math.trunc(time.time())
    ten_years_ago = datetime.datetime.now() - relativedelta(years=5)
    past_timestamp = math.trunc(time.mktime(ten_years_ago.timetuple()))
    request = requests.get('https://api.coingecko.com/api/v3/coins/ethereum/market_chart/range?vs_currency=usd&from=' + str(past_timestamp) + '&to=' + str(timestamp))
    data = request.json()
    if "prices" in data:
        formatted_data = []
        for value1, value2 in data["prices"]:
            date = datetime.datetime.fromtimestamp(value1 / 1000)
            formatted_data.append({"Date": date.strftime('%Y-%m-%d'), "Close": value2})
        with open('ethereum-data.json', 'w') as dumpfile:
            json.dump(formatted_data, dumpfile)

def get_sol_data():
    timestamp = math.trunc(time.time())
    ten_years_ago = datetime.datetime.now() - relativedelta(years=5)
    past_timestamp = math.trunc(time.mktime(ten_years_ago.timetuple()))
    request = requests.get(
        'https://api.coingecko.com/api/v3/coins/solana/market_chart/range?vs_currency=usd&from=' + str(
            past_timestamp) + '&to=' + str(timestamp))
    data = request.json()
    if "prices" in data:
        formatted_data = []
        for value1, value2 in data["prices"]:
            date = datetime.datetime.fromtimestamp(value1 / 1000)
            formatted_data.append({"Date": date.strftime('%Y-%m-%d'), "Close": value2})
        with open('solana-data.json', 'w') as dumpfile:
            json.dump(formatted_data, dumpfile)


get_btc_data()
get_eth_data()
get_sol_data()
print("Data Fetched")
