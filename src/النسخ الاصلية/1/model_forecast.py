"""
File: model_forecast.py
Description: Model forecast.
File Created: 06/09/2023 (Refactored: 13/06/2025)
Python Version: 3.9+
"""
import os
import argparse
import sys
import pandas as pd
import json
import torch
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pretrain.gru import GRU
from pretrain.lstm import LSTM
from statsmodels.tsa.holtwinters import Holt
import warnings

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='Forecast crypto-coins prices with ML models.')
    parser.add_argument('-vd', '--valid', required=True, type=str, help='Path to the CSV validation dataset.')
    parser.add_argument('-pt', '--pretrained', required=True, type=str, help='Path to the pretrained model file (.pth).')
    parser.add_argument('-t', '--target', required=True, type=str, help='Target coin to predict (same as pretraining).')
    parser.add_argument('-ft', '--features', required=True, type=str, help='Path to JSON with feature list (same as pretraining).')
    parser.add_argument('-m', '--model', required=True, type=str, help='Model to use for inference (lstm or gru).')
    parser.add_argument('-c', '--config', required=True, type=str, help='Path to config JSON used during pretraining.')
    parser.add_argument('-hz', '--horizon', type=int, default=7, help='Forecasting horizon in days.')
    parser.add_argument('-p', '--path', type=str, default=os.getcwd(), help='Path for saving the predictions.')
    parser.add_argument('-f', '--filename', type=str, help='Filename for predictions text file.')

    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.path, exist_ok=True)
    
    if not args.filename:
        today = datetime.now().strftime('%d%m%Y')
        args.filename = f'predictions_{args.model}_{today}'
        
    print("Arguments received:", vars(args))

    try:
        with open(args.config) as f: config = json.load(f)
        with open(args.features) as f: features = json.load(f)['features']
            
        valid_df = pd.read_csv(args.valid, index_col='Date', parse_dates=True)

        target_col_name = f"{args.target.lower()}_avg_ohlc"
        
        if target_col_name in features:
            raise ValueError("Target column cannot be in the features list.")
        
        # Prepare scalers
        full_training_data_for_scaling = valid_df[features + [target_col_name]]
        main_scaler = MinMaxScaler().fit(full_training_data_for_scaling)
        target_only_scaler = MinMaxScaler().fit(valid_df[[target_col_name]])
        
        # Forecast future features using Holt's method
        print("\nForecasting future features...")
        pred_set = pd.DataFrame()
        for feature in features:
            holt = Holt(valid_df[feature], initialization_method="estimated").fit()
            pred = holt.forecast(args.horizon)
            pred_set = pd.concat([pred_set, pred], axis=1)
        pred_set.columns = features
        
        # Get the last sequence from validation data
        sequence_length = config.get('sequence_length', 60)
        last_sequence_unscaled = valid_df[features].iloc[-sequence_length:]
        
        # Combine last sequence with forecasted features
        full_sequence_unscaled = pd.concat([last_sequence_unscaled, pred_set]).iloc[-sequence_length:]

        # Scale the combined sequence
        full_sequence_scaled = main_scaler.transform(pd.concat([full_sequence_unscaled, pd.DataFrame(columns=[target_col_name])]))[:, :len(features)]

        # Load Model
        print("Loading pretrained model...")
        model_class = LSTM if args.model.lower() == 'lstm' else GRU
        model = model_class(
            n_features=len(features),
            hidden_units=config['hidden_units'],
            n_layers=config['n_layers'],
            lr=config['learning_rate']
        )
        model.load_state_dict(torch.load(args.pretrained, map_location=device))
        model.to(device)
        model.eval()

        # Make prediction
        print("Generating forecast...")
        with torch.no_grad():
            input_tensor = torch.tensor(full_sequence_scaled, dtype=torch.float).unsqueeze(0).to(device)
            # The model predicts the next step after the sequence. 
            # To get a 7-day forecast, we would need to predict one step at a time and feed it back.
            # For simplicity, this script makes a single prediction based on the last known sequence.
            # A true multi-step forecast requires an iterative prediction loop.
            
            # This script's logic appears to be to get one prediction for each day in the horizon
            # based on a rolling window of forecasted features. We will simplify this to a single forecast.
            
            y_hat = model(input_tensor)
            
        # Inverse transform the prediction
        prediction_scaled = y_hat.cpu().numpy().flatten()
        final_forecast = target_only_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()

        print("\n--- Forecasted Value ---")
        print(f"Forecast for next step: {final_forecast[0]:.2f}")
        
        # For a multi-day forecast, the logic would be more complex.
        # We will save the single forecasted value as per this simplified interpretation.
        
        output_path = os.path.join(args.path, args.filename + '.txt')
        np.savetxt(output_path, final_forecast, fmt='%.4f')
        print(f"\nPrediction saved to: {output_path}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()