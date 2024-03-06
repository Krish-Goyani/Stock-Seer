# Stock Seer

Stock Seer is a project that predicts future stock prices using an LSTM (Long Short-Term Memory) model. It takes user input for a stock ticker symbol and the number of future sessions to predict, and then forecasts the stock's value for those sessions.

## Project Structure

The project consists of the following stages:

### 1. Data Ingestion

This stage handles the ingestion of stock data using yahoo finance API. It retrieves historical stock prices required for training the model.

### 2. Data Validation

The data validation stage ensures the integrity and quality of the ingested data. It it ensures thet our fetched data have required features.
### 3. Data Transformation

In this stage, the raw data is transformed and preprocessed to a format suitable for training the LSTM model. This  include feature scaling, and splitting the data into training and testing sets.

### 4. Partial Model Training

The partial model training stage trains the LSTM model on a train data.

### 5. Model Evaluation

After partial training, the model is evaluated on a held-out test set to assess its performance. This stage calculates relevant metrics, such as mean squared error and mean absolute percentage error, to gauge the model's accuracy.

### 6. Full Model Training

Once the model has been evaluated and optimized, the final stage trains the LSTM model on the complete dataset. This trained model is then used for making predictions on new, unseen stock data.

## Usage

To use Stock Seer, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone <repository_link>
   ```
2. **Install Anaconda:**
   
   Make sure you have Anaconda installed on your system. If not, you can download and install it from the official website: https://www.anaconda.com/download/
   
4. **Create a Virtual Environment:**
   
   Create a new virtual environment using Python 3.10:

   ```bash
   conda create --name your_env_name python=3.10 -y
   ```
   Replace your_env_name with the desired name for your virtual environment.
   
   Activate the newly created environment:
   ```bash
   conda activate your_env_name
   ```
5. **Install Dependencies:**
   
   Install the project dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
   This command will install all the required packages listed in the requirements.txt file.

6. **Run the Streamlit App:**
   ```bash
   streamlit run StockSeer.py
   ```
   This command will start the Streamlit app.
## Contributing

Contributions to Stock Seer are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.
