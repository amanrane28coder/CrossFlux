import logging
import certifi, os as _os
_os.environ.setdefault("SSL_CERT_FILE", certifi.where())
_os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
from tardis_dev import download_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_incremental")

def download_arbitrage_data():
    target_date = "2024-03-01" 
    
    exchanges = ["binance", "kraken"]
    symbols = ["BTCUSDT", "XBT/USD"]
    
    print(f"Initiating high-speed download for {target_date}...")
    
    for exchange in exchanges:
        symbol = symbols[0] if exchange == "binance" else symbols[1]
        logger.info(f"Downloading {exchange} {symbol}...")
        download_datasets(
            exchange=exchange,
            data_types=["incremental_book_L2", "trades"],
            from_date="2024-03-01",
            to_date="2024-03-02",
            symbols=[symbol],
            api_key="", # Leave empty for sample data
            download_dir="./data/raw"
        )
        print(f"Successfully staged {exchange} data in ./data/raw/")

if __name__ == "__main__":
    download_arbitrage_data()
