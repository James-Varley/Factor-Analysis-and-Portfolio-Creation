A Python framework for portfolio construction. Runs four different factor based models (CAPM, FF3, FF5, CH4) over five different observation windows (12, 24, 36, 48 and 60 months) and outputs the top 5 performing portfolios and summary statistics.

## Installation

To get started, install the dependancies using *pip*

``` pip install yfinance statsmodels tqdm ```

## Usage

To change the assets considered by the alogorithm, simply alter *line 23* to include any tickers *as they appear on the NASDAQ*

``` tickers = ['NVDA', 'MSFT', 'AAPL', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO'] ```

(as default the code considers the BATMMAAN stocks)



**This project is completed and the repository is archived**
