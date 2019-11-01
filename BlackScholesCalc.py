##Daniel Sims##
##2019##

#Because this script contains pyodbc and SQLAlchemy, it will not be
#debuggable on mac. The method for debugging is to remove those libraries
#and use the provided .csv file as input. From there the functions can be
#debugged and finally the libraries can be brought back into play.

## All information on the below functions can be found at ##
## https://aaronschlegel.me/black-scholes-formula-python.html ##
from __future__ import division # This is for Python 2.7 because of a divide by 0 error. 
import numpy as np
import scipy.stats as si
# import sympy as sy
# import sympy.statistics as systats
import pandas as pd
import pyodbc
import sqlalchemy
from sqlalchemy import create_engine
import datetime
import sys

def euro_vanilla_call(S, K, T, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call

def euro_vanilla_put(S, K, T, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    put = ((K * np.exp(-r * T)) * (si.norm.cdf(-d2, 0.0, 1.0)) - (S * si.norm.cdf(-d1, 0.0, 1.0)))
    
    return put

def euro_vanilla(S, K, T, r, sigma, option = 'call'):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option == 'call':
        result = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
        
    return result

# def euro_call_sym(S, K, T, r, sigma):  #Sympy Implementation of above functions
#
#     #S: spot price
#     #K: strike price
#     #T: time to maturity
#     #r: interest rate
#     #sigma: volatility of underlying asset
#
#     N = systats.Normal(0.0, 1.0)
#
#     d1 = (sy.ln(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
#     d2 = (sy.ln(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
#
#     call = (S * N.cdf(d1) - K * sy.exp(-r * T) * N.cdf(d2))
#
#     return call

# def euro_put_sym(S, K, T, r, sigma):
#
#     #S: spot price
#     #K: strike price
#     #T: time to maturity
#     #r: interest rate
#     #sigma: volatility of underlying asset
#
#     N = systats.Normal(0.0, 1.0)
#
#     d1 = (sy.ln(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
#     d2 = (sy.ln(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
#
#     put = (K * sy.exp(-r * T) * N.cdf(-d2) - S * N.cdf(-d1))
#
#     return put

# def sym_euro_vanilla(S, K, T, r, sigma, option = 'call'):
#
#     #S: spot price
#     #K: strike price
#     #T: time to maturity
#     #r: interest rate
#     #sigma: volatility of underlying asset
#
#     N = systats.Normal(0.0, 1.0)
#
#     d1 = (sy.ln(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
#     d2 = (sy.ln(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
#
#     if option == 'call':
#         result = (S * N.cdf(d1) - K * sy.exp(-r * T) * N.cdf(d2))
#     if option == 'put':
#         result = (K * sy.exp(-r * T) * N.cdf(-d2) - S * N.cdf(-d1))
#
#     return result


####################################################################
#######################Dividend Paying##############################
####################################################################

def black_scholes_call_div(S, K, T, r, q, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #q: rate of continuous dividend paying asset 
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    call = (S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call

def black_scholes_put_div(S, K, T, r, q, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #q: rate of continuous dividend paying asset 
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0))
    
    return put

def euro_vanilla_dividend(S, K, T, r, q, sigma, option = 'call'):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #q: rate of continuous dividend paying asset 
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if option == 'call':
        result = (S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0))
        
    return result

# def black_scholes_call_div_sym(S, K, T, r, q, sigma):
#
#     #S: spot price
#     #K: strike price
#     #T: time to maturity
#     #r: interest rate
#     #q: rate of continuous dividend paying asset
#     #sigma: volatility of underlying asset
#
#     N = systats.Normal(0.0, 1.0)
#
#     d1 = (sy.ln(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
#     d2 = (sy.ln(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
#
#     call = S * sy.exp(-q * T) * N.cdf(d1) - K * sy.exp(-r * T) * N.cdf(d2)
#
#     return call

# def black_scholes_call_put_sym(S, K, T, r, q, sigma):
#
#     #S: spot price
#     #K: strike price
#     #T: time to maturity
#     #r: interest rate
#     #q: rate of continuous dividend paying asset
#     #sigma: volatility of underlying asset
#
#     N = systats.Normal(0.0, 1.0)
#
#     d1 = (sy.ln(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
#     d2 = (sy.ln(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
#
#     put = K * sy.exp(-r * T) * N.cdf(-d2) - S * sy.exp(-q * T) * N.cdf(-d1)
#
#     return put

# def sym_euro_vanilla_dividend(S, K, T, r, q, sigma, option = 'call'):
#
#     #S: spot price
#     #K: strike price
#     #T: time to maturity
#     #r: interest rate
#     #q: rate of continuous dividend paying asset
#     #sigma: volatility of underlying asset
#
#     N = systats.Normal(0.0, 1.0)
#
#     d1 = (sy.ln(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
#     d2 = (sy.ln(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * sy.sqrt(T))
#
#     if option == 'call':
#         result = S * sy.exp(-q * T) * N.cdf(d1) - K * sy.exp(-r * T) * N.cdf(d2)
#     if option == 'put':
#         result = K * sy.exp(-r * T) * N.cdf(-d2) - S * sy.exp(-q * T) * N.cdf(-d1)
#
#     return result


def euro_delta_call(S, K, T, r, sigma):

    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    return si.norm.cdf(d1)

def euro_delta_put(S, K, T, r, sigma):

    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    return -(si.norm.cdf(-d1))


####################################################################
#####################Implied Volatility#############################
####################################################################

def newton_vol_call(S, K, T, C, r, sigma):
    # S: spot price
    # K: strike price
    # T: time to maturity
    # C: Call value
    # r: interest rate
    # sigma: volatility of underlying asset

    d1 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    fx = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0) - C

    vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)

    tolerance = 0.000001
    x0 = sigma
    xnew = x0
    xold = x0 - 1

    while abs(xnew - xold) > tolerance:
        xold = xnew
        xnew = (xnew - fx - C) / vega

        return abs(xnew)


def newton_vol_put(S, K, T, P, r, sigma):
    d1 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    fx = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0) - P

    vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)

    tolerance = 0.000001
    x0 = sigma
    xnew = x0
    xold = x0 - 1

    while abs(xnew - xold) > tolerance:
        xold = xnew
        xnew = (xnew - fx - P) / vega

        return abs(xnew)


# def newton_vol_call_div(S, K, T, C, r, q, sigma):
#     d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#
#     fx = S * np.exp(-q * T) * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0) - C
#
#     vega = (1 / np.sqrt(2 * np.pi)) * S * np.exp(-q * T) * np.sqrt(T) * np.exp((-si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)
#
#     tolerance = 0.000001
#     x0 = sigma
#     xnew = x0
#     xold = x0 - 1
#
#     while abs(xnew - xold) > tolerance:
#         xold = xnew
#         xnew = (xnew - fx - C) / vega
#
#         return abs(xnew)
#
#
# def newton_vol_put_div(S, K, T, P, r, q, sigma):
#     d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#
#     fx = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * si.norm.cdf(-d1, 0.0, 1.0) - P
#
#     vega = (1 / np.sqrt(2 * np.pi)) * S * np.exp(-q * T) * np.sqrt(T) * np.exp((-si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)
#
#     tolerance = 0.000001
#     x0 = sigma
#     xnew = x0
#     xold = x0 - 1
#
#     while abs(xnew - xold) > tolerance:
#         xold = xnew
#         xnew = (xnew - fx - P) / vega
#
#         return abs(xnew)


####################################################################
###########################Vega#####################################
####################################################################

def vega(S, K, T, r, sigma):
    # S: spot price
    # K: strike price
    # T: time to maturity
    # r: interest rate
    # sigma: volatility of underlying asset

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    vega = S * si.norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)

    return vega/100 #Divide by 100 because Vega measures the price response to a 100% move in Vol.


def main():
    print("Hello Black, Hello Scholes. Oh hai Merton.")

    if len(sys.argv) == 1: 
        mode = 'Default'
    elif len(sys.argv) == 2 and sys.argv[1] == 'Full':
        mode = 'Full'
    elif len(sys.argv) == 2 and sys.argv[1] == 'Incremental':
        mode = 'Incremental'
    else:
        print("Acceptable arguments are Full or Incremental")
        sys.exit(0)
    
    server = 'server-name.us-east-1.rds.amazonaws.com' 
    database = 'database-name' 
    username = 'user' 
    password = 'pass'

    riskFreeRate = .021 # July 24, 2019
    
    # For reading from SQL Server
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
    
    # For writing to SQL Server
    engine = create_engine('mssql+pyodbc://test:user@server-name.us-east-1.rds.amazonaws.com/Database-name?driver=SQL+Server')


    if mode == 'Full' or mode == 'Default':
        sql = "SELECT ID, Time, Expires, CITMAsk, CATMAsk, PITMAsk, PATMAsk, Price, RoundedPlusPrice, RoundedMinusPrice, HV30 FROM Athena WHERE HV30 is not NULL"
    elif mode == 'Incremental':
        maxsql = "Select MAX(ID) FROM BlackScholes"
        maxID = pd.read_sql(maxsql,cnxn)
        maxIDstr = maxID.values.max().astype('str')
        sql = "SELECT * FROM Athena WHERE ID > " + maxIDstr
    
        
    data = pd.read_sql(sql,cnxn)

    # Leaving this here for troubleshooting (on Mac) if necessary
    # data = pd.read_csv("Athena_Full_Select.csv")
    # data

    # Do a little bit of refactoring. Might or might not be necessary, actually.
    data["Time"] = pd.to_datetime(data["Time"])
    data["Expires"] = pd.to_datetime(data["Expires"])

    # Calculate fraction of year using the expiration date.
    data["Fraction of Year"] = (data["Expires"] + datetime.timedelta(days=1) - data["Time"]).dt.total_seconds() / (86400 * 365)

    #Calculate IV - Within 'data' - which is strangely inefficient, but needs to be done here.

    data['CATM IV'] = 0
    data['CITM IV'] = 0
    data['PATM IV'] = 0
    data['PITM IV'] = 0

    #Take note that the price here is ASK - For initial simplicity.
    for i, row in data.iterrows():
        data.loc[i, 'CATM IV'] = newton_vol_call(row["Price"], row["Price"], row["Fraction of Year"], row["CATMAsk"],
                                                 riskFreeRate, row["HV30"])
        data.loc[i, 'CITM IV'] = newton_vol_call(row["Price"], row["RoundedMinusPrice"], row["Fraction of Year"],
                                                 row["CITMAsk"], riskFreeRate, row["HV30"])
        data.loc[i, 'PATM IV'] = newton_vol_put(row["Price"], row["Price"], row["Fraction of Year"], row["PATMAsk"],
                                                riskFreeRate, row["HV30"])
        data.loc[i, 'PITM IV'] = newton_vol_put(row["Price"], row["RoundedPlusPrice"], row["Fraction of Year"],
                                                row["PITMAsk"], riskFreeRate, row["HV30"])

    data

    # data['CATM BS'] = euro_vanilla_call(data["Price"], data["Price"], data["Fraction of Year"], 0.025, data["HV30"])
    # data['CATM Delta'] = euro_delta_call(data["Price"], data["Price"], data["Fraction of Year"], 0.025, data["HV30"])
    # data['CITM BS'] = euro_vanilla_call(data["Price"], data["RoundedMinusPrice"], data["Fraction of Year"], 0.025, data["HV30"])
    # data['CITM Delta'] = euro_delta_call(data["Price"], data["RoundedMinusPrice"], data["Fraction of Year"], 0.025, data["HV30"])
    #
    # data['PATM BS'] = euro_vanilla_put(data["Price"], data["Price"], data["Fraction of Year"], 0.025, data["HV30"])
    # data['PATM Delta'] = euro_delta_put(data["Price"], data["Price"], data["Fraction of Year"], 0.025, data["HV30"])
    # data['PITM BS'] = euro_vanilla_put(data["Price"], data["RoundedPlusPrice"], data["Fraction of Year"], 0.025, data["HV30"])
    # data['PITM Delta'] = euro_delta_put(data["Price"], data["RoundedPlusPrice"], data["Fraction of Year"], 0.025, data["HV30"])
    #
    # data['C Spread BS'] = data['CITM BS'] - data['CATM BS']
    # data['P Spread BS'] = data['PITM BS'] - data['PATM BS']


    output = pd.DataFrame()

    output['ID'] = data['ID']
    output['Expiration Fraction'] = data['Fraction of Year']

    # CATM
    output['CATM BS'] = euro_vanilla_call(data["Price"], data["Price"], data["Fraction of Year"], riskFreeRate, data["HV30"])
    output['CATM Delta'] = euro_delta_call(data["Price"], data["Price"], data["Fraction of Year"], riskFreeRate, data["HV30"])
    output['CATM IV'] = data['CATM IV']
    output['CATM Vega'] = vega(data["Price"], data["Price"], data["Fraction of Year"], riskFreeRate, data["HV30"])

    # CITM
    output['CITM BS'] = euro_vanilla_call(data["Price"], data["RoundedMinusPrice"], data["Fraction of Year"], riskFreeRate,
                                        data["HV30"])
    output['CITM Delta'] = euro_delta_call(data["Price"], data["RoundedMinusPrice"], data["Fraction of Year"], riskFreeRate,
                                         data["HV30"])
    output['CITM IV'] = data['CITM IV']
    output['CITM Vega'] = vega(data["Price"], data["RoundedMinusPrice"], data["Fraction of Year"], riskFreeRate,
                                         data["HV30"])

    # PATM
    output['PATM BS'] = euro_vanilla_put(data["Price"], data["Price"], data["Fraction of Year"], riskFreeRate, data["HV30"])
    output['PATM Delta'] = euro_delta_put(data["Price"], data["Price"], data["Fraction of Year"], riskFreeRate, data["HV30"])
    output['PATM IV'] = data['PATM IV']
    output['PATM Vega'] = vega(data["Price"], data["Price"], data["Fraction of Year"], riskFreeRate, data["HV30"])

    # PITM
    output['PITM BS'] = euro_vanilla_put(data["Price"], data["RoundedPlusPrice"], data["Fraction of Year"], riskFreeRate,
                                       data["HV30"])
    output['PITM Delta'] = euro_delta_put(data["Price"], data["RoundedPlusPrice"], data["Fraction of Year"], riskFreeRate,
                                        data["HV30"])
    output['PITM IV'] = data['PITM IV']
    output['PITM Vega'] = vega(data["Price"], data["RoundedPlusPrice"], data["Fraction of Year"], riskFreeRate,
                                        data["HV30"])

    output['C Spread BS'] = output['CITM BS'] - output['CATM BS']
    output['P Spread BS'] = output['PITM BS'] - output['PATM BS']
    
    if mode == 'Full' or mode == 'Default': # Replace the whole table
        output.to_sql('BlackScholes', engine, if_exists='replace', index=False)
    elif mode == 'Incremental': # Add new calculations to existing table
        output.to_sql('BlackScholes', engine, if_exists='append', index=False)

    # output.to_csv('JoinResults.csv')
    # data.to_csv('BlackScholesResults.csv')

    print("Done")

main()
