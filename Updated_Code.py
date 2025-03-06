import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import ollama
import tempfile
import base64
import os
from prophet import Prophet
import requests

def get_weather(city_state, api_key):
    try:
        # Normalize the input by stripping extra spaces and capitalizing properly
        city_state = city_state.strip().title().replace(" ", ", ")
        
        # Define a dictionary for state abbreviations
        state_abbr = {
            "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
            "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
            "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
            "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
            "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
            "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
            "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
            "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
            "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
            "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
        }

        # Check if a state abbreviation or full name is provided
        for full_state, abbr in state_abbr.items():
            if full_state.lower() in city_state.lower():
                city_state = city_state.replace(full_state, abbr)
                break
        
        # Call the OpenWeatherMap API with the normalized input
        response = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather?q={city_state}&appid={api_key}&units=imperial"
        )
        
        if response.status_code == 200:
            data = response.json()
            weather_desc = data["weather"][0]["description"].capitalize()
            temp = data["main"]["temp"]
            return f"{city_state}: {weather_desc}, {temp}°F"
        else:
            return "Unable to fetch weather data. Please check your entered city and state."
    except Exception as e:
        return f"Error fetching weather data: {e}"

api_key = '161a02f55af96a556113c9c2379c3f69'

# Streamlit app setup
st.set_page_config(layout="wide")
st.title("StockForecastX: An AI Stock Forecasting Application")

# Weather input
st.sidebar.header("Weather Settings")
weather_city = st.sidebar.text_input("Enter City for Weather Info:", "Denver")

# Fetch and display weather
weather_info = get_weather(weather_city, api_key)
st.markdown(f"### 🌤 Weather Update: {weather_info}", unsafe_allow_html=True)

# Warning notice
if "warning_shown" not in st.session_state:
    st.session_state["warning_shown"] = False

if not st.session_state["warning_shown"]:
    st.warning("⚠️ Please note: This application utilizes AI-powered stock analysis and forecasting. While the AI provides valuable insights, the predictions may not always be accurate and should not be relied upon for real-world trading decisions.")
    st.session_state["warning_shown"] = True

# Sidebar inputs for stock forecasting
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL):", "AAPL")
forecast_period = st.sidebar.slider("Forecast Days", min_value=1, max_value=30, value=7)
start_date = st.sidebar.date_input("Select Start Date", pd.to_datetime("2024-01-01"))
end_date = pd.to_datetime("today")
st.sidebar.write(f"Forecasting starting from {start_date}")

# Fetch stock data
if st.sidebar.button("Fetch Data"):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            st.error("No data found for the specified ticker and date range.")
        else:
            st.session_state["stock_data"] = stock_data
            st.success("Stock data loaded successfully!")
            show_ai_prediction = True  # This ensures the AI prediction part appears after data is fetched
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# Display raw stock data if available
if "stock_data" in st.session_state:
    data = st.session_state["stock_data"]
    st.subheader("Raw Stock Data")
    st.write(data)

    # Prepare data for forecasting with Prophet
    df = data.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]  # Rename columns for Prophet compatibility

    # Fit the Prophet model and make predictions
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)

    # Display forecast table
    forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-forecast_period:]
    forecast_table.columns = ["Date", "Predicted Price", "Lower Bound", "Upper Bound"]
    st.subheader("Stock Price Forecast")
    st.dataframe(forecast_table)

    # Provide CSV download link
    csv = forecast_table.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  
    href = f'<a href="data:file/csv;base64,{b64}" download="forecast.csv">Download Forecast as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Show the next day prediction
    next_day_prediction = forecast.iloc[-forecast_period]
    st.sidebar.write(f"Next Day Prediction: {next_day_prediction['yhat']:.2f}")

# Display stock data and process forecasting
if "stock_data" in st.session_state:
    data = st.session_state["stock_data"]
    df = data.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]
    
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)

    # Prepare the graph
    fig = go.Figure()
    
    # Add VWAP
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP', line=dict(color='blue')))
    
    # Add Forecasted Price
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Price', line=dict(dash='dot', color='orange')))
    
    # Add Moving Averages
    for period, color in zip([9, 21, 50, 200], ['purple', 'green', 'red', 'black']):
        data[f'EMA_{period}'] = data['Close'].ewm(span=period).mean()
        fig.add_trace(go.Scatter(x=data.index, y=data[f'EMA_{period}'], mode='lines', name=f'EMA {period}', line=dict(color=color)))
    
    # Bollinger Bands
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['STD_20'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['SMA_20'] + (2 * data['STD_20'])
    data['BB_Lower'] = data['SMA_20'] - (2 * data['STD_20'])
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines', name='Bollinger Upper', line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines', name='Bollinger Lower', line=dict(color='gray')))
    
    # Display graph
    fig.update_layout(title=f"Stock Analysis for {ticker}", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig)

# Streamlit setup for AI Stock Prediction
if 'show_ai_prediction' in locals() and show_ai_prediction:
    st.title("Stock Price Prediction Explanation with AI")

    # Fetch stock data and make predictions
    if st.button("Predict Stock Price"):
        with st.spinner("Fetching stock data and generating prediction..."):
            try:
                # Fetch stock data
                stock_data = yf.download(ticker, period="1y")
                
                # Prepare data for Prophet
                df = stock_data.reset_index()[["Date", "Close"]]
                df.columns = ["ds", "y"]  # Rename columns for Prophet compatibility
                
                # Fit Prophet model and make predictions
                model = Prophet()
                model.fit(df)
                future = model.make_future_dataframe(periods=forecast_period)
                forecast = model.predict(future)
                
                # Get predicted stock price
                predicted_price = forecast["yhat"].iloc[-1]
                
                # Display prediction
                st.write(f"The predicted price for {ticker} in the next {forecast_period} days is: ${predicted_price:.2f}")
                
                # Generate explanation for prediction using Ollama
                ai_prompt = f"""
                You are a financial analyst. Explain the factors that could influence the stock price of {ticker} based on the forecasted price.
                The predicted price for {ticker} is ${predicted_price:.2f}.
                Consider trends, moving averages, and other relevant financial factors in your explanation.
                """

                # Use Ollama for analysis
                try:
                    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": ai_prompt}])
                    # Display explanation
                    st.write("**AI Explanation:**")
                    st.write(response["message"]["content"])
                except Exception as e:
                    st.error(f"Error communicating with Ollama: {e}")

            except Exception as e:
                st.error(f"Error predicting stock price: {e}")
