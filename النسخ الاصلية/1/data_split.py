"""
File: data_split.py
Description: Train-valid split for model pre-training with advanced features including volatility.
File Created: 06/09/2023 (Modified: 13/06/2025 by Gemini - Final Version)
Python Version: 3.9+
"""

import os
import argparse
from datetime import datetime
import sys
import pandas as pd
import pandas_ta as ta
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    parser = argparse.ArgumentParser(description='Split coin price dataset and generate features')
    parser.add_argument('-d', '--data', type=str, required=True, help='path to the csv dataset')
    parser.add_argument('-tr', '--train', type=float, required=True, help='ratio for train set split')
    parser.add_argument('-vd', '--valid', type=float, required=True, help='ratio for valid set split')
    parser.add_argument('-t', '--target', type=str, required=True, help='target coin symbol, e.g., BTC')
    parser.add_argument('-p', '--path', type=str, default=os.getcwd(), help='path for saving the splits (default is current directory)')
    parser.add_argument('-f', '--filenames', nargs='+', help='filenames for dataset splits (defaults are train_TODAY, etc)')

    args = parser.parse_args()

    if not os.path.exists(args.path):
        print('Invalid path provided: destination does not exist!')
        sys.exit(1)
    if sum([args.train, args.valid]) != 1:
        print('Invalid arguments provided: sum of train and valid ratios is not 1!')
        sys.exit(1)

    target_coin_upper = args.target.upper()

    if not args.filenames:
        now = datetime.now()
        today = now.strftime('%d%m%Y')
        filename_1 = f'train_{today}'
        filename_2 = f'valid_{today}'
    else:
        filename_1, filename_2 = args.filenames[0], args.filenames[1]

    print(f"Starting data processing for target: {target_coin_upper}")

    try:
        data = pd.read_csv(args.data, sep=',')
        data.drop_duplicates(subset=['Date', 'Coin'], keep='last', inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])

        print("Pivoting data to wide format...")
        pivoted_df = data.pivot(index='Date', columns='Coin')

        all_features_list = []
        
        print("Calculating technical indicators for all coins...")
        for coin in data['Coin'].unique():
            print(f"... calculating for {coin}")
            
            coin_df = pd.DataFrame({
                'open': pivoted_df.get(('Open', coin)), 'high': pivoted_df.get(('High', coin)),
                'low': pivoted_df.get(('Low', coin)), 'close': pivoted_df.get(('Close', coin)),
                'volume': pivoted_df.get(('Volume', coin)),
            }).copy()
            
            if coin_df.isnull().all().all():
                print(f"... skipping {coin} due to all missing data.")
                continue

            # حساب وإضافة المؤشرات بشكل آمن وصريح
            sma = coin_df.ta.sma(length=7)
            rsi = coin_df.ta.rsi(length=14)
            atr = coin_df.ta.atr(length=14)
            macd = coin_df.ta.macd()
            bbands = coin_df.ta.bbands(length=20)
            
            coin_df[f"{coin.lower()}_sma7"] = sma
            coin_df[f"{coin.lower()}_rsi14"] = rsi
            coin_df[f"{coin.lower()}_atr"] = atr
            if macd is not None and not macd.empty:
                coin_df[[f"{coin.lower()}_macd", f"{coin.lower()}_macdh", f"{coin.lower()}_macds"]] = macd
            if bbands is not None and not bbands.empty:
                coin_df[[f"{coin.lower()}_bbl", f"{coin.lower()}_bbm", f"{coin.lower()}_bbu", f"{coin.lower()}_bbb", f"{coin.lower()}_bbp"]] = bbands
            
            coin_df[coin.upper()] = coin_df['close']
            coin_df[f"{coin.lower()}_volume"] = coin_df['volume']
            
            coin_df.drop(columns=['open', 'high', 'low', 'close', 'volume'], inplace=True)
            
            all_features_list.append(coin_df)

        final_features_df = pd.concat(all_features_list, axis=1)

        # --- هذه هي الخطوة الهامة التي تعالج المشكلة ---
        target_new_name = f"{target_coin_upper.lower()}_avg_ohlc"
        final_features_df.rename(columns={target_coin_upper: target_new_name}, inplace=True)
        
        print("Handling NaN values...")
        final_features_df.bfill(inplace=True)
        final_features_df.ffill(inplace=True)

        df = final_features_df.reset_index()

        print("Splitting data into train and validation sets...")
        train_size = int(len(df) * args.train)
        train = df.iloc[:train_size].copy()
        valid = df.iloc[train_size:].copy()

        train.set_index('Date', inplace=True)
        valid.set_index('Date', inplace=True)

        file_train = os.path.join(args.path, filename_1 + '.csv')
        train.to_csv(file_train, sep=',', encoding='utf-8', index=True)
        print(f"Train set saved to {file_train}")

        file_valid = os.path.join(args.path, filename_2 + '.csv')
        valid.to_csv(file_valid, sep=',', encoding='utf-8', index=True)
        print(f"Validation set saved to {file_valid}")

    except Exception as e:
        print(f"An unexpected error occurred in data_split: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()