import streamlit as st
import pandas as pd
import yfinance as yf
import cufflinks as cf
import datetime 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from joblib import load
import time
from yfinance import exceptions as yf_exceptions
from requests.exceptions import HTTPError

# Helper function to fetch ticker info with retry logic
def get_ticker_info_safe(ticker, max_retries=2, delay=1):
    """Safely fetch ticker info with retry logic and error handling"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait = delay * (attempt + 1)
                time.sleep(wait)
            # Try get_info() first (yfinance 0.2.58+), fallback to .info
            if hasattr(ticker, 'get_info'):
                info = ticker.get_info()
            else:
                info = ticker.info
            if info and isinstance(info, dict) and len(info) > 0:
                return info
            return {}
        except (yf_exceptions.YFRateLimitError, HTTPError) as e:
            # Handle rate limiting specifically
            if getattr(e, 'response', None) is not None and getattr(e.response, 'status_code', None) == 429:
                wait = delay * (attempt + 1)
                st.warning(f"Yahoo Finance rate limit reached. Retrying in {wait} seconds...")
                time.sleep(wait)
                continue
            if attempt == max_retries - 1:
                st.error("Yahoo Finance rate limit exceeded. Please wait and try again.")
                return {}
        except yf_exceptions.YFTickerMissingError:
            st.error("Ticker data is not available or the ticker may be delisted.")
            return {}
        except Exception as e:  # Fallback for other unexpected errors
            if attempt == max_retries - 1:
                st.error(f"Unexpected error loading stock information: {e}")
                return {}
    return {}

def get_fast_metrics(ticker):
    """Return fast_info data without breaking the app when unavailable."""
    try:
        fast = ticker.fast_info
        if fast and isinstance(fast, dict):
            return fast
    except Exception:
        pass
    return {}

#app title
st.markdown("""
# Stock Price App
Shown are the Stock Price of the selected stock.

**Credits**
- App Built By Harshit Varshney (0901CS233D07) & Lokendra Sharma (0901CS233D08)
- Built in Python using Streamlit, cufflinks, pandas, datetime, plotly and yfinance libraries.
""")

st.write("--------")
#sidebar
st.sidebar.subheader("Select Stock")
start_date=st.sidebar.date_input("Start Date",datetime.date(2019,1,1))
end_date=st.sidebar.date_input("End Date",datetime.date(2024,12,1))

#Getting Ticker Data
ticker_list=pd.read_csv('NSE.csv')
ticker_list_LSE=pd.read_csv('LSE.csv')
tickerlist2=pd.read_csv('nasdaq-listed.csv')
Market=st.sidebar.selectbox("Select Market",["NSE","NASDAQ","LSE"])
if Market=="NSE":
    ticker_Symbol=st.sidebar.selectbox("NSE Stock Symbol ",ticker_list['SYMBOL'])
    ticker_Symbol=ticker_Symbol+".NS"
elif Market=="LSE":
    ticker_Symbol=st.sidebar.selectbox("LSE Stock Symbol",ticker_list_LSE['Symbol'])
    ticker_Symbol=ticker_Symbol+".L"

    #ticker_Symbol=st.sidebar.selectbox("Stock Symbol",crypto_list["CODE"])

    #tickerData=yf.Ticker(f"{ticker_Symbol}-USD")
else:
    ticker_Symbol=st.sidebar.selectbox("Stock Symbol",tickerlist2)

# Create ticker object first
tickerData = yf.Ticker(ticker_Symbol)
# Use start and end dates from sidebar, don't specify period
tickerDf = tickerData.history(start=start_date, end=end_date)
    
# Initialize session state for caching
if 'ticker_info_cache' not in st.session_state:
    st.session_state.ticker_info_cache = {}

if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = None

now = datetime.datetime.now()
cache_entry = st.session_state.ticker_info_cache.get(ticker_Symbol)
fetch_new_info = True
if cache_entry and st.session_state.last_ticker == ticker_Symbol:
    age = (now - cache_entry['fetched_at']).total_seconds()
    if age < 300:  # reuse cached info for 5 minutes
        fetch_new_info = False

# Fetch ticker info with caching to avoid rate limiting
if fetch_new_info:
    with st.spinner('Fetching stock information...'):
        data = get_ticker_info_safe(tickerData, max_retries=2, delay=2)
        st.session_state.ticker_info_cache[ticker_Symbol] = {
            'data': data,
            'fetched_at': now
        }
        st.session_state.last_ticker = ticker_Symbol
else:
    data = cache_entry['data']

fast_metrics = get_fast_metrics(tickerData)

if Market=="NSE" or Market=="NASDAQ":
    if st.sidebar.button("Predict Stock") and data is not None:
        stock=ticker_Symbol
        try:
            if Market=="NASDAQ":
                model_dir=f"./models/{stock}/"
                scaler1 = load(f'{model_dir}{stock}_scaler.joblib')
                model1 = load(f'{model_dir}{stock}_predictor.joblib')
            else:
                # For NSE, ticker_Symbol already has .NS suffix, so use it directly
                model_dir=f"./modelsNS/{stock}/"
                scaler1 = load(f'{model_dir}{stock}_scaler.joblib')
                model1 = load(f'{model_dir}{stock}_predictor.joblib')
        except FileNotFoundError:
            st.error(f"⚠️ Prediction model not found for {stock}")
            st.info(f"Model files are missing in: {model_dir}\nPrediction is only available for pre-trained stocks.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading prediction model: {e}")
            st.stop()
            
        current_date = datetime.datetime.now()
        # Load saved model and scaler
        
        try:
            SmaEma=tickerData.history(period="max")
        except yf_exceptions.YFRateLimitError:
            st.error("Yahoo Finance rate limit hit while loading historical data. Please wait a moment and try again.")
            st.stop()
        except Exception as err:
            st.error(f"Unable to load historical data for {stock}: {err}")
            st.stop()
        

        open_price = fast_metrics.get('open') or data.get('open') if isinstance(data, dict) else None
        current_price = fast_metrics.get('last_price') or data.get('currentPrice') if isinstance(data, dict) else None
        day_high = fast_metrics.get('day_high') or data.get('dayHigh') if isinstance(data, dict) else None
        day_low = fast_metrics.get('day_low') or data.get('dayLow') if isinstance(data, dict) else None
        volume_value = fast_metrics.get('volume') or data.get('volume') if isinstance(data, dict) else None

        required_values = [open_price, current_price, day_high, day_low, volume_value]
        if any(v is None for v in required_values):
            st.error("Latest price data is unavailable for this ticker right now. Please refresh or try again later.")
            st.stop()

        open_close=open_price-current_price
        high_low=day_high-day_low
        volume=volume_value
        quarter_end= 1 if current_date.month % 3 == 0 else 0
        current_date=datetime.datetime.today().date()
        today_data = pd.DataFrame({
            'Open': open_price,
            'High': day_high,
            'Low': day_low,
            'Close': current_price,
            'Volume': volume_value
        }, index=[current_date])  # Ensure the index matches ticker.history


        SmaEma=pd.concat([SmaEma,today_data])
        # st.sidebar.write(SmaEma.tail())

        sma10=SmaEma['Close'].rolling(window=10).mean()[-1]
        sma50=SmaEma['Close'].rolling(window=50).mean()[-1]
        sma200=SmaEma['Close'].rolling(window=200).mean()[-1]
        ema10=SmaEma['Close'].ewm(span=10,adjust=False).mean()[-1]
        ema50=SmaEma['Close'].ewm(span=50,adjust=False).mean()[-1]
        ema200=SmaEma['Close'].ewm(span=200,adjust=False).mean()[-1]
        print(open_close, high_low, volume, quarter_end,sma10,sma50,sma200,ema10,ema50,ema200)
        new_data = [[open_close, high_low, volume, quarter_end,sma10,sma50,sma200,ema10,ema50,ema200]]  # Replace with actual feature values
        new_data_scaled = scaler1.transform(new_data)
        
        prediction = model1.predict(new_data_scaled)
        probability = model1.predict_proba(new_data_scaled)
        if prediction[0] == 1 :
            color="green"
            st.sidebar.markdown(f"<h2 style='color:{color}'>Bullish<br>Probability of Bullish: {round(probability[0][1]*100,2)}%</h2>",unsafe_allow_html=True)
            
        else:
            color='red'
            st.sidebar.markdown(f"<h2 style='color:{color}'>Bearish<br> Probability of Bearish: {round(probability[0][1]*100,2)}%</h3>",unsafe_allow_html=True)
            
if ticker_Symbol:
    # Add refresh button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh"):
            # Clear cache to force refresh
            st.session_state.ticker_info_cache = {}
            st.session_state.last_ticker = None
            st.rerun()
    
    #Ticker Information
    try:
        ticker_info = data if isinstance(data, dict) else {}
        if not ticker_info and not fast_metrics:
            st.warning("Yahoo Finance did not return data for this ticker. It may be temporarily unavailable or delisted.")
        stock_name=ticker_info.get('longName') or ticker_info.get('shortName') or ticker_Symbol
        st.write(f"# {stock_name}")

        current_price = fast_metrics.get('last_price') or ticker_info.get('currentPrice')
        previous_close = fast_metrics.get('previous_close') or ticker_info.get('previousClose')
        
        if current_price is not None and previous_close is not None and previous_close != 0:
            change=round(current_price - previous_close, 2)
            changeper=round(change/(previous_close/100), 2)
            if change>0:
                color='green'
            elif change<0:
                color='red'
            else:
                color='grey'
            currency = fast_metrics.get('currency') or ticker_info.get('currency', 'USD')
            st.markdown(f"<h2 style='color:{color}'>{current_price} {currency} <h3 style='color:{color}'>{change} {changeper}%</h3></h2>",unsafe_allow_html=True)
        else:
            st.info("Real-time price data is currently unavailable for this ticker.")

        company_website = ticker_info.get('website', '')
        if company_website:
            domain = company_website.split("//")[-1].split("/")[0]  # Extract domain
            logo_url = f"https://logo.clearbit.com/{domain}"
            try:
                st.image(logo_url)
            except:
                pass

        if 'sector' in ticker_info:
            st.write(f"**Sector :** {ticker_info['sector']}")
        if st.pills("",options="View Long Summary"):
            if 'longBusinessSummary' in ticker_info:
                st.write(ticker_info['longBusinessSummary'])
        st.write("Fundamentals :")
        if 'marketCap' in ticker_info:
            st.write(f"**Market Cap :** {ticker_info['marketCap']} {ticker_info.get('currency', 'USD')}")
        if 'enterpriseValue' in ticker_info:
            st.write(f"**Enterprise Value :** {ticker_info['enterpriseValue']} {ticker_info.get('currency', 'USD')}")
        if 'floatShares' in ticker_info:
            st.write(f"**Float Shares :** {ticker_info['floatShares']}")
    except Exception as e:
        st.error(f"Error loading stock information: {str(e)}")
        st.info("Some stock information may be unavailable. Please try another stock or check your connection.")


    periods=['1d','5d','1mo','2mo']
    #candlestick Chart 
    if ticker_Symbol:
        if not tickerDf.empty:
            st.write("--------")
            
            cperiod=st.selectbox("Select Date range",periods)
            # Use period only, ignore start/end dates for the chart
            tickerDf=tickerData.history(period=cperiod)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3],
                        subplot_titles=['Candlestick Chart', 'Volume Chart'],
                        row_titles=['', 'Volume'])
            fig.add_trace(go.Candlestick(
                            x=tickerDf.index,
                            open=tickerDf['Open'],
                            high=tickerDf['High'],
                            low=tickerDf['Low'],
                            close=tickerDf['Close'],
                            name="Candlestick"
                            ), row=1, col=1)
            fig.add_trace(
                    go.Bar(
                        x=tickerDf.index,
                        y=tickerDf['Volume'],
                        name='Volume',
                        marker_color='rgba(0, 200, 0, 0.5)'  
                    ),
                    row=2, col=1  )

            #Chart Layout
            fig.update_layout(
                dragmode="zoom",
                xaxis=dict(
                rangeslider=dict(visible=True),  # Enable the range slider for scrolling
                type="date"),
                yaxis=dict(
                fixedrange=False  ),
                title=f'{ticker_Symbol} Stock Price and Volume',
                legend=dict(
                orientation="h",  # Horizontal orientation
                yanchor="top",
                y=-0.2,  # Move legend below the chart
                xanchor="center",
                x=0.5),
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                xaxis_rangeslider_visible=False,  
                height=600  
            )
            #display the chart
            st.plotly_chart(fig,use_container_width=True)
    #bollinger band
    if not tickerDf.empty:
        st.write("--------")
        bperiod=st.selectbox("Select Date range for Bollinger Bands",periods)
        MA=st.selectbox("Select DMA For Bollinger Bands",[10,20,30,40,50,60])
        # Use period only for Bollinger Bands
        tickerDf=tickerData.history(period=bperiod)
        #Moving Average Calculation
        tickerDf[f'{MA}_MA'] = tickerDf['Close'].rolling(window=MA).mean()
        tickerDf['Upper_Band'] = tickerDf[f'{MA}_MA'] + 2 * tickerDf['Close'].rolling(window=MA).std()
        tickerDf['Lower_Band'] = tickerDf[f'{MA}_MA'] - 2 * tickerDf['Close'].rolling(window=MA).std()
        # Plotting Bollinger Bands
        fig2 = go.Figure(data=[
        go.Candlestick(x=tickerDf.index,
                    open=tickerDf['Open'],
                    high=tickerDf['High'],
                    low=tickerDf['Low'],
                    close=tickerDf['Close'],
                    name='Candlestick'),
        go.Scatter(x=tickerDf.index, y=tickerDf['Upper_Band'], mode='lines', name='Upper Bollinger Band'),
        go.Scatter(x=tickerDf.index, y=tickerDf['Lower_Band'], mode='lines', name='Lower Bollinger Band'),
        go.Scatter(x=tickerDf.index, y=tickerDf[f'{MA}_MA'], mode='lines', name=f'{MA} Day Moving Average')
    ])
        fig2.update_layout(title=f"{ticker_Symbol} Bollinger Band",
            dragmode="zoom",
            xaxis=dict(
            rangeslider=dict(visible=True),  # Enable the range slider for scrolling
            type="date"),
            yaxis=dict(
            fixedrange=False),
            legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="top",
            y=-0.2,  # Move legend below the chart
            xanchor="center",
            x=0.5
        ),xaxis_title="Date",yaxis_title="Price",xaxis_rangeslider_visible=False)
        st.plotly_chart(fig2)