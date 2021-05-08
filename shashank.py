import numpy as np
import pandas as pd
from math import exp
import matplotlib.pyplot as plt
from american_option_pricing import AmericanOptionPricing

#def get_expected_divs(ticker, t, T, dividends):
#    out = dividends.loc[(dividends.index >t) & (dividends.index <T) & (dividends.TICKER == ticker),:]
#    out['days'] = (out.index - t).days
#    return out

# Compute value of call options according to monte carlo method
def price_american_options(options, stocks, verbose = True):
    count = 0
    own_vol_calc = []
    implied_calc = []
    for row in options:
        count = count +1
        row = row[1]
        S = row['Price']
        K = row['strike_price']/1000
        #vol = 0.25
        rf = row['rf']
        mu = rf
        T = row['T']
        ticker = row['ticker']
        price = row['midpoint']
        q = row['q']
        data =pd.DataFrame({'Close':stocks[ticker][stocks.index<=row['date']]})
        data.index = data.index.rename('Date')
        #divs = get_expected_divs(ticker,row['date'],pd.to_datetime(row['exdate']))

        a = AmericanOptionPricing(ticker,data,pd.to_datetime(row['date']),pd.to_datetime(row['exdate']),K,q)
        a.risk_free_rate = rf
        cp = 0 if row['cp_flag'] == "C" else 1
      
        
        ov = a.calculate_option_prices()[cp]
        own_vol_calc.append(ov)
        a.volatility = row['impl_volatility']
        iv = a.calculate_option_prices()[cp]
        implied_calc.append(iv)
        
        if verbose:
            print("\n------------------------------------------------------------------------------------------")
            print("{i}: {ticker} {cp} with strike {K} and days to maturity {T} with spot {S} at {date}".format(
            i = count, ticker = ticker, cp = "call" if cp ==0 else "put", K = K, T=T, S = round(S,2),date = row['date']))
            print("------------------------------------------------------------------------------------------")
            print("With calculated vol of {vol} :  {c}".format(vol = round(a.volatility,3), c = round(ov,2)))
            print("With an implied vol of {vol} :  {c}".format(vol = round(a.volatility,3), c = round(iv,2)))
            print("Actual observed call midpoint:  " + str(round(row['midpoint'],2)))
            
    return own_vol_calc, implied_calc
