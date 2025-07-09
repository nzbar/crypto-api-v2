"""
File: pretrain.py
Description: Model pretrain.
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
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pretrain.gru import GRU
from pretrain.lstm import LSTM
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

class StockDataset(Dataset):
    """Custom PyTorch Dataset for creating sequences."""
    def __init__(self, data, target_col, feature_cols, sequence_length):
        self.features = data[feature_cols].values
        self.target = data[target_col].values
        self.sequence_length = sequence_length

    def __len__(self):
        return self.target.shape[0] - self.sequence_length + 1

    def __getitem__(self, index):
        features_sequence = self.features[index:index + self.sequence_length]
        target_value = self.target[index + self.sequence_length - 1]
        return torch.tensor(features_sequence, dtype=torch.float), torch.tensor(target_value, dtype=torch.float)

def main():
    parser = argparse.ArgumentParser(description='Pretrain ML models for crypto-coins forecast')
    parser.add_argument('--train', type=str, required=True, help='Path to the CSV training dataset.')
    parser.add_argument('--valid', type=str, required=True, help='Path to the CSV validation dataset.')
    parser.add_argument('--target', type=str, required=True, help='Target coin to predict (e.g., BTC).')
    parser.add_argument('--features', type=str, required=True, help='Path to JSON file with feature list.')
    parser.add_argument('--model', type=str, required=True, help='Model to train (lstm or gru).')
    parser.add_argument('--config', type=str, required=True, help='Path to JSON file with config for pretraining.')
    parser.add_argument('--path', type=str, default=os.getcwd(), help='Path for saving the pretrained model.')
    parser.add_argument('--filename', type=str, help='Filename for the model.')
    
    args = parser.parse_args()
    
    gpus_available = 1 if torch.cuda.is_available() else 0
    os.makedirs(os.path.join(args.path, 'models'), exist_ok=True)
    
    if not args.filename:
        today = datetime.now().strftime('%d%m%Y')
        args.filename = f'{args.model}_{today}'

    print("Arguments received:", vars(args))

    try:
        with open(args.config) as f: config = json.load(f)
        with open(args.features) as f: features = json.load(f)['features']
        
        train_df = pd.read_csv(args.train, index_col='Date', parse_dates=True)
        valid_df = pd.read_csv(args.valid, index_col='Date', parse_dates=True)

        target_col_name = f"{args.target.lower()}_avg_ohlc"

        if target_col_name in features:
            raise ValueError("Invalid data format: Target column cannot be in the features list.")
        
        if target_col_name not in train_df.columns:
            raise ValueError(f"Target column '{target_col_name}' not found in the dataset.")

        missing_features = [f for f in features if f not in train_df.columns]
        if missing_features:
            raise ValueError(f"The following features are not in the dataset: {missing_features}")

        target_and_features = features + [target_col_name]
        train_subset = train_df[target_and_features]
        valid_subset = valid_df[target_and_features]
        
        scaler = MinMaxScaler()
        train_scaled = pd.DataFrame(scaler.fit_transform(train_subset), index=train_subset.index, columns=train_subset.columns)
        valid_scaled = pd.DataFrame(scaler.transform(valid_subset), index=valid_subset.index, columns=valid_subset.columns)

        model_type = args.model.lower()
        if model_type not in ['gru', 'lstm']:
            raise ValueError("Invalid model type. Choose 'gru' or 'lstm'.")
            
        pl.seed_everything(config['seed'])
        
        sequence_length = config.get('sequence_length', 60)
        train_dataset = StockDataset(train_scaled, target_col=target_col_name, feature_cols=features, sequence_length=sequence_length)
        validation_dataset = StockDataset(valid_scaled, target_col=target_col_name, feature_cols=features, sequence_length=sequence_length)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'])

        early_stopping = EarlyStopping('val_loss', patience=config['patience'])
        
        model_class = LSTM if model_type == 'lstm' else GRU
        model = model_class(
            n_features=len(features),
            hidden_units=config['hidden_units'],
            n_layers=config['n_layers'],
            lr=config['learning_rate']
        )
        
        # --- الشرح: تم تعديل هذا الجزء لحل المشكلة ---
        trainer = pl.Trainer(
            callbacks=[early_stopping],
            max_epochs=config['n_epochs'],
            accelerator='gpu' if gpus_available else 'cpu',
            devices=1 # المكتبة تتوقع الرقم 1 دائمًا هنا سواء للمعالج المركزي أو الرسومي
        )
        
        print(f"\nStarting training for {model_type.upper()} model...")
        trainer.fit(model, train_loader, validation_loader)
        
        output_path = os.path.join(args.path, 'models', args.filename + '.pth')
        torch.save(model.state_dict(), output_path)
        print(f"\nTraining complete. Model saved to: {output_path}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()