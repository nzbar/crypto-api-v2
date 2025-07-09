"""
File: data_pull.py
Description: Dataset generator from CoinGecko API.
File Created: 06/09/2023 (Refactored: 13/06/2025)
Python Version: 3.9+
"""
import os
import json
import argparse
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
import time

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
COINGECKO_IDS = {
    'btc': 'bitcoin', 'eth': 'ethereum', 'usdt': 'tether', 'usdc': 'usd-coin',
    'bnb': 'binancecoin', 'xrp': 'ripple', 'busd': 'binance-usd', 'ada': 'cardano',
    'sol': 'solana', 'doge': 'dogecoin', 'dot': 'polkadot', 'dai': 'dai',
    'shib': 'shiba-inu', 'trx': 'tron', 'avax': 'avalanche-2', 'uni': 'uniswap',
    'wbtc': 'wrapped-bitcoin', 'leo': 'leo-token', 'ltc': 'litecoin'
}

def fetch_crypto_data_from_coingecko(coin_symbol, start_date_str, end_date_str):
    coin_id = COINGECKO_IDS.get(coin_symbol.lower())
    if not coin_id:
        print(f"Error: CoinGecko ID for '{coin_symbol}' not found. Please add it to the COINGECKO_IDS dictionary.")
        return None

    print(f"Fetching data for {coin_symbol.upper()} from {start_date_str} to {end_date_str} using CoinGecko...")

    try:
        start_dt_obj = datetime.strptime(start_date_str, '%d-%m-%Y').replace(tzinfo=timezone.utc)
        end_dt_for_api = datetime.strptime(end_date_str, '%d-%m-%Y').replace(tzinfo=timezone.utc) + timedelta(days=1)

        start_timestamp = int(start_dt_obj.timestamp())
        end_timestamp = int(end_dt_for_api.timestamp())

        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart/range"
        params = {'vs_currency': 'usd', 'from': start_timestamp, 'to': end_timestamp}

        time.sleep(13)  # Be polite to the API
        response = requests.get(url, params=params)
        response.raise_for_status()
        json_data = response.json()

        prices_data = json_data.get('prices', [])
        volumes_data = json_data.get('total_volumes', [])
        
        if not prices_data or not volumes_data:
            print(f"No data returned for {coin_symbol.upper()}.")
            return None

        df_prices = pd.DataFrame(prices_data, columns=['timestamp', 'Close'])
        df_volumes = pd.DataFrame(volumes_data, columns=['timestamp', 'Volume'])

        df_prices['Date'] = pd.to_datetime(df_prices['timestamp'], unit='ms').dt.date
        df_volumes['Date'] = pd.to_datetime(df_volumes['timestamp'], unit='ms').dt.date

        df_merged = pd.merge(df_prices[['Date', 'Close']], df_volumes[['Date', 'Volume']], on='Date', how='inner')
        df_merged['Open'] = df_merged['High'] = df_merged['Low'] = df_merged['Close']
        df_merged['Coin'] = coin_symbol.upper()
        
        # Filter to exact date range
        df_merged['Date'] = pd.to_datetime(df_merged['Date'])
        start_dt_obj_naive = datetime.strptime(start_date_str, '%d-%m-%Y')
        end_dt_obj_naive = datetime.strptime(end_date_str, '%d-%m-%Y')
        df_merged = df_merged[(df_merged['Date'] >= start_dt_obj_naive) & (df_merged['Date'] <= end_dt_obj_naive)]

        return df_merged[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Coin']]

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {coin_symbol.upper()}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for {coin_symbol.upper()}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate cryptocurrency OHLCV dataset from CoinGecko.')
    parser.add_argument('--path', type=str, default=".", help='Output folder path [default: current directory].')
    parser.add_argument('--filename', type=str, help='Output filename [default: dataset_coingecko_YYYYMMDD_YYYYMMDD.csv].')
    parser.add_argument('--coins', type=str, default='btc,eth,usdt,usdc,bnb,xrp,busd,ada,sol,doge,dot,dai,shib,trx,avax,uni,wbtc,leo,ltc', help='Comma-separated list of crypto symbols.')
    parser.add_argument('--start', type=str, default=(datetime.now() - timedelta(days=365)).strftime('%d-%m-%Y'), help='Start date (DD-MM-YYYY) [default: one year ago].')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%d-%m-%Y'), help='End date (DD-MM-YYYY) [default: today].')
    
    args = parser.parse_args()

    # Create path if it doesn't exist
    os.makedirs(args.path, exist_ok=True)
    
    # Generate filename if not provided
    if not args.filename:
        start_fn = datetime.strptime(args.start, '%d-%m-%Y').strftime('%Y%m%d')
        end_fn = datetime.strptime(args.end, '%d-%m-%Y').strftime('%Y%m%d')
        args.filename = f'dataset_coingecko_{start_fn}_{end_fn}.csv'
    
    cryptos = [coin.strip() for coin in args.coins.split(',')]
    all_dfs = []

    print(f"Starting data pull for {len(cryptos)} coins from {args.start} to {args.end}...")
    for crypto in cryptos:
        df = fetch_crypto_data_from_coingecko(crypto, args.start, args.end)
        if df is not None and not df.empty:
            all_dfs.append(df)
            
    if not all_dfs:
        print("No data was fetched for any coin. Exiting.")
        sys.exit(1)
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df['Date'] = final_df['Date'].dt.strftime('%Y-%m-%d')
    final_df = final_df[['Coin', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    output_path = os.path.join(args.path, args.filename)
    final_df.to_csv(output_path, sep=',', encoding='utf-8', index=False)
    print(f"\nDataset saved successfully to: {output_path}")

if __name__ == '__main__':
    main()