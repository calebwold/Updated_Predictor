Overview
StockForecastX is an AI-powered stock forecasting application that uses machine learning and time-series analysis to predict stock prices. This app integrates weather information, stock data, and AI-powered explanations for stock price predictions. The predictions and analysis are made using Prophet, a forecasting tool, and Ollama, an AI model that provides detailed explanations for stock movements.

⚠️ Important Note: AI Feature Requires Ollama
The AI-based stock price explanation feature in this application relies on Ollama, a machine learning-powered conversational AI model. Without Ollama, the AI will not be able to generate explanations for stock predictions.

🛠️ How to Set Up Ollama for AI Predictions
To enable the AI features, follow these steps to download, install, and set up Ollama:

1. Download Ollama
Go to Ollama's official website and download the version compatible with your operating system:

Windows
macOS
Linux
2. Install Ollama
Follow the installation instructions for your specific OS. You may need to authenticate or set up a user account depending on the platform.

3. Verify Ollama Installation
To ensure Ollama is properly installed, open your terminal (or command prompt) and run the following command:

bash
Copy
Edit
ollama models
This will list the available models and confirm that Ollama is set up correctly.

4. Install the Ollama Python Package
After installing Ollama, you need the Python client to integrate it into your project. Install it using pip:

bash
Copy
Edit
pip install ollama
5. Test the AI Prediction Feature
Once Ollama is set up and the Python package is installed, the AI stock price prediction feature will work as intended. The AI model will generate explanations based on the forecasted stock prices.

🚀 How the Application Works
Stock Data Fetching: The app fetches real-time stock data using yFinance and allows users to input a stock ticker symbol (e.g., AAPL) and a date range.

Forecasting with Prophet: The Prophet library is used to predict future stock prices based on historical data.

Weather Updates: The app also integrates weather data via OpenWeatherMap API to provide users with current weather information for a selected city.

AI-Powered Explanation: Once the stock data and forecast are ready, Ollama's AI model generates an explanation of the factors influencing the stock prediction, including trends, moving averages, and financial analysis.

❓ Why Ollama is Required
AI Model Interaction: Ollama’s conversational AI model, such as llama3, is required to generate insights and explanations about stock price predictions.
Real-Time Data Analysis: Ollama uses AI to analyze the factors behind stock movements and provide explanations based on forecasted data.
Without Ollama installed, the AI-based explanation feature will not function. The app will still display stock predictions, but without the AI’s analysis.

