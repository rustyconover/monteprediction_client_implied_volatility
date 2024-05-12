#!/bin/env python3
#
# Monteprediction.com entry -
#
# Author: Rusty Conover <rusty@conover.me>
#
# Goal: predict the weekly returns of the 11 SPDR ETFs across 1 million scenarios.
#
# Method:
# 1. construct the implied volatilities for each ETF then simulate from that
#    IV a week's worth of returns.
#
# Caveats:
#
# 1. Not all SPDR ETFs have liquid options pricing, so the IVs may be inaccurate, do some work
#    to attempt to clean them up.
# 2. IV is overstated compared to realized volatility, because options market makers mark up risk,
#    nothing is done yet to attempt to correct this yet.

import numpy as np
import pandas as pd
import os
import yfinance as yf
import scipy.interpolate
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from datetime import datetime, timedelta


from monteprediction import SPDR_ETFS
from monteprediction.submission import send_in_chunks

# Tournament settings, don't change these.
num_samples_per_chunk = int(1048576 / 8)
num_chunks = 8
num_samples = num_chunks * num_samples_per_chunk


def simulate_stock_returns(sigma: float) -> float:
    """Generate simulated stock returns for a given volatility.

    Parameters
    ----------
    sigma : float
        Volatitlity

    Returns
    -------
    float
        Returns
    """
    r = 0.05  # Risk-free interest rate
    T = 1  # Time period in years
    days_to_simulate = 5  # Number of simulations
    num_days = 252  # Number of trading days in a year
    dt = T / num_days

    # So this is just going to do a draw on those returns.
    S = [1]

    root = np.sqrt(dt)

    # I know this is really slow, but its left as an exercise to the reader to improve it.
    for i in range(days_to_simulate):
        Z = np.random.normal(size=1)
        S *= np.exp(np.cumsum((r - 0.5 * sigma**2) * dt + sigma * root * Z))

    return S[0]


def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def error_function(sigma, option_price, S, K, T, r):
    return (black_scholes_call(S, K, T, r, sigma) - option_price) ** 2


def implied_volatility(option_price, S, K, T, r):
    # Initial guess for volatility
    initial_guess = 0.5

    # Using scipy minimize function to find the minimum of the error function
    result = minimize(error_function, initial_guess, args=(option_price, S, K, T, r))

    return result.x[0]


def get_atm_implied_volatility(symbol: str, expiration_date: str, price: float):
    option_chain = yf.Ticker(symbol).option_chain(expiration_date)

    # Extract call option data, don't use the puts here because they will
    # be affected by skew.
    calls = option_chain.calls

    # It seems that Yahoo Finance implied volatility calculated at the
    # midpoint between the buy/ask spread, but since this program may
    # not be run when the market is open, it makes more sense to
    # calculate it at the last price.

    # Remove strikes that didn't trade.
    calls = calls[calls["volume"] > 2]

    # Only deal with strikes that have traded in the last three days.
    calls = calls[
        calls["lastTradeDate"]
        >= (datetime.now().date() - timedelta(days=3)).strftime("%Y-%m-%d")
    ]

    recalculated_ivs = []

    # Recalculate the IV from the last traded price, rather than the midpoint of the bid/ask
    for row in calls.itertuples():
        option_price = row.lastPrice
        S = price
        K = row.strike
        T = (
            (
                datetime.date(datetime.strptime(expiration_date, "%Y-%m-%d"))
                - datetime.date(row.lastTradeDate)
            ).days
        ) / 365  # Time to expiration in years (30 days remaining)
        r = 0.05  # Risk-free interest rate

        implied_vol = implied_volatility(option_price, S, K, T, r)
        recalculated_ivs.append(implied_vol)

    calls["last_trade_iv"] = recalculated_ivs

    # If there are enough strikes try to interpolate the different
    # strikes in a volatility smile, otherwise just be naive and average them.
    if len(calls) > 3:
        atm_iv_calls = scipy.interpolate.BSpline(
            calls["strike"], calls["last_trade_iv"], 1
        )(price)
    else:
        atm_iv_calls = np.mean(calls["last_trade_iv"])

    return atm_iv_calls


results = {}
for symbol in SPDR_ETFS:
    # Not all SPDR ETFs have the same expiration, just take the average of the closest two for now.
    # Ideally we'd take the most liquid option expiration chain, but there could be
    # short term events that bring more IV forward.
    all_ivs = []
    options_expiration_dates = yf.Ticker(symbol).options

    # Get the last close so the ATM price can be determined.
    price = yf.Ticker(symbol).history(period="1d")["Close"].iloc[-1]

    for expiration_index in range(1, 3):
        expiration_date = options_expiration_dates[expiration_index]
        atm_iv = get_atm_implied_volatility(symbol, expiration_date, price)
        all_ivs.append(atm_iv)

    # Take the average of the IVs for the two different expiration dates,
    # near term options have more juice, so possibly smooth this out.
    atm_iv = np.mean(all_ivs)

    results[symbol] = [1 - simulate_stock_returns(atm_iv) for i in range(num_samples)]

df = pd.DataFrame(results)

assert len(df.index) == num_samples, f"Expecting exactly {num_samples} samples"
assert list(df.columns) == SPDR_ETFS, "Columns should match SPDR_ETFS in order"

response = send_in_chunks(
    df, num_chunks=num_chunks, email=os.environ["EMAIL"], name=os.environ["NAME"]
)
print(response)
