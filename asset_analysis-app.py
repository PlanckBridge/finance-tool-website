import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from scipy.stats import skew, kurtosis
from datetime import datetime

import streamlit as st

# Display the introduction message
st.markdown("""
# **About This Tool**

This tool allows you to analyze and compare the returns of financial assets. All calculations are based on **daily adjusted closing prices**, fetched directly from Yahoo Finance.

### **Enter Tickers**
Input the **Yahoo Finance tickers** of the assets you want to analyze (examples: `AAPL` for Apple, `BTC-USD` for Bitcoin in USD, or `^SP500TR` for the S&P 500 Total Return Index).  
Separate multiple tickers with commas.

### **What You'll Get**
- Metrics such as:
  - **CAGR** (Compound Annual Growth Rate)
  - **Volatility** (the standard deviation of daily returns, annualized)
  - **Maximum Drawdown** (the worst decline during the analysis period)
  - **Skewness and Kurtosis** of daily returns
  - **Cumulative Returns** (based on daily adjusted closing prices)
- A **correlation matrix** for selected assets
- **Cumulative returns visualized over time**
""")

PRICE_COLUMNS = ['Adj Close', 'Close']

def fetch_stock_data(tickers):
    """Fetch historical data for multiple tickers."""
    try:
        data = yf.download(tickers, group_by='ticker', period="max", threads=True)
        if data.empty:
            st.error(f"No data found for tickers: {', '.join(tickers)}")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            return {ticker: data[ticker].dropna() for ticker in tickers}
        return {tickers[0]: data.dropna()}
    except Exception as e:
        st.error(f"Error fetching data for tickers: {tickers}: {e}")
        return None

def align_stock_data(data_dict, option, custom_start_date=None, custom_end_date=None):
    """Align data based on the chosen alignment option."""
    try:
        if option == "Analyze from the earliest common date in the dataset":
            common_start_date = max(data.index.min() for data in data_dict.values())
            data_dict = {ticker: data[data.index >= common_start_date] for ticker, data in data_dict.items()}
        elif option == "Analyze since a custom date" and custom_start_date and custom_end_date:
            data_dict = {
                ticker: data[(data.index >= custom_start_date) & (data.index <= custom_end_date)]
                for ticker, data in data_dict.items()
            }
        data_dict = {ticker: data for ticker, data in data_dict.items() if not data.empty}
        if not data_dict:
            raise ValueError("All datasets are empty after alignment.")
    except ValueError as ve:
        st.error(f"Alignment error: {ve}")
        return {}
    return data_dict

def calculate_statistics(data, price_column):
    """Calculate financial statistics for a stock."""
    try:
        if len(data) < 2:
            return {}

        start_date = data.index.min().date()
        end_date = data.index.max().date()
        years = (data.index.max() - data.index.min()).days / 365.25
        final_value = data['Cumulative Return'].iloc[-1]
        cagr = (final_value + 1) ** (1 / years) - 1
        rolling_max = data[price_column].cummax()
        drawdowns = (data[price_column] - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        daily_returns = data[price_column].pct_change().dropna()
        if len(daily_returns) < 2:
            return {}
        skewness = skew(daily_returns)
        excess_kurtosis = kurtosis(daily_returns, fisher=True)
        volatility = daily_returns.std() * (252 ** 0.5)
        return {
            "Start Date": str(start_date),
            "End Date": str(end_date),
            "Years": round(years, 4),
            "CAGR (%)": round(cagr * 100, 4),
            "Max Drawdown (%)": round(max_drawdown * 100, 4),
            "Volatility (%)": round(volatility * 100, 4),
            "Skewness": round(skewness, 4),
            "Excess Kurtosis": round(excess_kurtosis, 4),
            "Cumulative Return (%)": round(final_value * 100, 4)
        }
    except Exception as e:
        st.error(f"Error calculating statistics: {e}")
        return {}

def calculate_correlation_table(data_dict, sorted_tickers):
    """Calculate and return a correlation matrix for the tickers, sorted by cumulative return."""
    # Collect daily returns for each ticker
    daily_returns = pd.DataFrame({
        ticker: data['Adj Close'].pct_change().dropna()
        for ticker, data in data_dict.items() if 'Adj Close' in data.columns
    })
    
    # Compute correlation matrix
    correlation_matrix = daily_returns.corr()
    
    # Reorder rows and columns based on sorted_tickers
    correlation_matrix = correlation_matrix.loc[sorted_tickers, sorted_tickers]
    
    # Add "Correlation" in the top-left cell
    correlation_matrix.index.name = "Correlation Matrix"  # Set the index name to appear in the top-left corner
    
    # Round values for readability
    correlation_matrix = correlation_matrix.round(4)
    return correlation_matrix

def plot_cumulative_returns(data_dict, stats_dict, sorted_tickers, title):
    """Plot cumulative returns for multiple securities with dynamic Y-axis formatting."""
    fig = go.Figure()

    # Determine time horizon (in days) based on the data range
    time_horizon_days = (max(data.index.max() for data in data_dict.values()) -
                         min(data.index.min() for data in data_dict.values())).days

    # Choose tick format dynamically: 2 decimals for short time horizons, 0 decimals for long
    tickformat = ".2%" if time_horizon_days < 30 else ".0%"

    for ticker in sorted_tickers:
        data = data_dict[ticker]
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Cumulative Return'],
            mode='lines',
            name=f"{ticker} ({stats_dict[ticker]['Cumulative Return (%)']:.2f}%)"
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Return",
        yaxis=dict(tickformat=tickformat),  # Dynamic tick format
        legend_title="Assets",
        margin=dict(l=20, r=20, t=40, b=20),  # Adjust margins for better spacing
    )
    st.plotly_chart(fig)

def fetch_ticker_info(tickers):
    """Fetch the full name and description for each ticker."""
    info = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info[ticker] = stock.info.get('longName', 'No description available')
        except Exception as e:
            info[ticker] = f"Error fetching data: {e}"
    return info

st.title("Asset Returns Analysis")
tickers = st.text_input("Enter a ticker or tickers separated by commas (ex: AAPL or AAPL,MSFT,BTC-USD)").split(',')
tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
if tickers:
    with st.spinner("Fetching stock data..."):
        data_dict = fetch_stock_data(tickers)
    if data_dict:
        align_option = st.radio(
            "Choose data alignment option:",
            ["Analyze all data available", "Analyze from the earliest common date in the dataset", "Analyze since a custom date"]
        )
        custom_start_date = None
        if align_option == "Analyze since a custom date":
            col1, col2 = st.columns(2)
            
            with col1:
                custom_start_date = st.date_input(
                    "Select start date:",
                    value=datetime(datetime.now().year, 1, 1).date(),
                    min_value=datetime(1900, 1, 1).date(),
                    max_value=datetime.now().date(),
                )
            
            with col2:
                custom_end_date = st.date_input(
                    "Select end date:",
                    value=datetime.now().date(),
                    min_value=custom_start_date,  # End date must be after the start date
                    max_value=datetime.now().date(),
                )
            
            # Pass both dates to align the data
            data_dict = align_stock_data(data_dict, align_option, pd.to_datetime(custom_start_date), pd.to_datetime(custom_end_date))

        elif align_option != "Analyze all data available":
            data_dict = align_stock_data(data_dict, align_option)

        if data_dict:
            stats_dict = {}
            for ticker, data in data_dict.items():
                price_col = next((col for col in PRICE_COLUMNS if col in data.columns), None)
                if price_col:
                    data['Cumulative Return'] = data[price_col] / data[price_col].iloc[0] - 1
                    stats_dict[ticker] = calculate_statistics(data, price_col)

            # Sort tickers by cumulative return
            sorted_tickers = sorted(
                stats_dict.keys(),
                key=lambda x: stats_dict[x]["Cumulative Return (%)"],
                reverse=True
            )

            # Create the stats DataFrame and transpose it for better readability
            stats_df = pd.DataFrame(stats_dict).T.loc[sorted_tickers]
            stats_df = stats_df.round(4)  # Round numerical values to 4 decimal places
            st.subheader("Statistics Summary")
            st.dataframe(stats_df.T, use_container_width=True)  # Transpose for tickers as headers

            # Calculate and display correlation matrix if there are multiple tickers
            if len(data_dict) > 1:
                st.subheader("Correlation Matrix")
                correlation_matrix = calculate_correlation_table(data_dict, sorted_tickers)
                
                # Ensure proper formatting and enable scrolling
                formatted_correlation_matrix = correlation_matrix.round(4)  # Format numbers to 4 decimals
                st.dataframe(
                    formatted_correlation_matrix,  # Display the DataFrame directly
                    use_container_width=True  # Enable dynamic resizing and scrolling
                )

            st.subheader("Cumulative Returns")
            plot_cumulative_returns(data_dict, stats_dict, sorted_tickers, "Cumulative Returns")

            # Display ticker descriptions in sorted order
            st.subheader("Ticker Descriptions")
            ticker_info = fetch_ticker_info(sorted_tickers)  # Use sorted_tickers for the order
            for ticker in sorted_tickers:
                description = ticker_info.get(ticker, "No description available")
                st.markdown(f"**{ticker}:** {description}")
