import streamlit as st
import yfinance as yf
from prophet import Prophet
from datetime import date
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Setting the start date and end date
start_date = "2015-01-01"
end_date = date.today().strftime("%Y-%m-%d")

# this will load stock data with caching
@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    # Download data and reset index to ensure 'Date' is a column
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)  # Ensure 'Date' is a column
    return data



# this is the function I used to simulate stock prices using Monte Carlo
def monte_carlo_simulation(data, days=365, num_simulations=100):
    returns = data['Adj Close'].pct_change().dropna()
    mean = returns.mean()
    std_dev = returns.std()
    simulations = np.zeros((num_simulations, days))
    last_price = data['Adj Close'].iloc[-1]

    for i in range(num_simulations):
        daily_returns = np.random.normal(mean, std_dev, days)
        price_series = [last_price]
        for r in daily_returns:
            new_price = price_series[-1] * (1 + r)
            price_series.append(new_price)
        simulations[i] = price_series[1:]

    return simulations

#this will be the title 
st.title("Stock Price Prediction with Monte Carlo Simulation")

stock_options = ["AAPL", "GOOGL", "MSFT", "GME", "AMZN", "META", "TSLA", "DIS", "NFLX", "NVDA"]

selected_stock = st.selectbox("Select a Stock", stock_options)

# Select the number of years to predict
num_years = st.slider("Number of Years for Prediction", 1, 4, value=1)
prediction_days = num_years * 365

# Load the stock data
st.write("Loading stock data...")
stock_data = load_stock_data(selected_stock, start_date, end_date)
st.write("Data loaded.")

# This code will Display the raw data in a Table
st.subheader ('Raw data')
st.write(stock_data.tail())

# Brief definitions of key stock metrics
st.subheader('Definitions of Stock Metrics')
st.write("**Open:** The stock price at the start of the trading day.")
st.write("**Close:** The stock price at the end of the trading day.")
st.write("**High:** The highest stock price during the trading day.")
st.write("**Low:** The lowest stock price during the trading day.")
st.write("**Adj Close:** The adjusted closing price, accounting for dividends and splits.")
st.write("**Volume:** The total number of shares traded during the day.")

# Plotting raw data with all key metrics
st.subheader("Detailed Stock Price History")
fig = go.Figure()
# Plot Open, High, Low, Close, Adj Close, Volume
fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Open'], mode='lines', name='Open'))
fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['High'], mode='lines', name='High'))
fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Low'], mode='lines', name='Low'))
fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Adj Close'], mode='lines', name='Adj Close'))

# Adding a secondary y-axis for volume to keep scale appropriate
fig.add_trace(go.Bar(x=stock_data['Date'], y=stock_data['Volume'], name='Volume', yaxis='y2'))

# Setting the layout with dual axes for price and volume
fig.update_layout(
    title="Stock Price History with Volume",
    yaxis=dict(title='Price(USD)', side='left'),
    yaxis2=dict(title='Volume', side='right', overlaying='y', showgrid=False),
    xaxis=dict(title='Date')
)
st.plotly_chart(fig)

# Monte Carlo simulation
st.subheader("Monte Carlo Simulation")
st.write("Running Monte Carlo simulation...")
simulations = monte_carlo_simulation(stock_data, days=prediction_days)

# Plotting the Monte Carlo simulations
fig = go.Figure()
for sim in simulations:
    fig.add_trace(go.Scatter(x=list(range(1, prediction_days + 1)), y=sim, mode='lines', showlegend=False))
st.plotly_chart(fig)

# Investment calculation
st.subheader("Investment Prediction")
investment_amount = st.slider("Enter an Investment Amount", 50, 10000, 500)

# Calculate the average simulation results for 1 week, 1 month, 6 months, and 1 year
simulated_mean = np.mean(simulations, axis=0)

# Ensure indices do not exceed the array's bounds
max_days = len(simulated_mean)
# Use min to avoid IndexError
profit_1_week = investment_amount * (simulated_mean[min(6, max_days - 1)] - stock_data['Adj Close'].iloc[-1])
profit_1_month = investment_amount * (simulated_mean[min(30, max_days - 1)] - stock_data['Adj Close'].iloc[-1])
profit_6_months = investment_amount * (simulated_mean[min(180, max_days - 1)] - stock_data['Adj Close'].iloc[-1])
profit_1_year = investment_amount * (simulated_mean[min(364, max_days - 1)] - stock_data['Adj Close'].iloc[-1])

st.write(f"1 Week: ${profit_1_week:.2f}")
st.write(f"1 Month: ${profit_1_month:.2f}")
st.write(f"6 Months: ${profit_6_months:.2f}")
st.write(f"1 Year: ${profit_1_year:.2f}")

# Addding a short note below the investment prediction
st.write("**Note:** These projections are based on simulations and are for informational purposes only. Actual investment returns are not guaranteed.")
