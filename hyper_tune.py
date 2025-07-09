"""
File: hyper_tune.py
Description: Hyperparameter tuning for LSTM model using Optuna.
File Created: 11/06/2025
Python Version: 3.9+
"""

# Imports
import os
import argparse
import sys
import pandas as pd
import json
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import optuna
from sklearn.preprocessing import MinMaxScaler # <-- السطر الجديد الذي تم إضافته

# استيراد الكلاسات من مشروعك
from pretrain.datasets import DatasetV1
from pretrain.lstm import LSTM

# دالة الهدف التي سيقوم Optuna بتحسينها
def objective(trial, args):
    """
    دالة الهدف لـ Optuna.
    تقوم بتدريب النموذج بإعدادات مقترحة وترجع قيمة الخسارة للتحقق.
    """
    # 1. اقتراح المعلمات الفائقة (Hyperparameters)
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'n_layers': trial.suggest_int('n_layers', 2, 8),
        'hidden_units': trial.suggest_int('hidden_units', 64, 256),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
    }
    
    print(f"\n--- بدء المحاولة رقم: {trial.number} ---")
    print(f"الإعدادات المقترحة: {params}")

    # 2. تحميل البيانات
    train_df = pd.read_csv(args.train, index_col='Date', parse_dates=True)
    valid_df = pd.read_csv(args.valid, index_col='Date', parse_dates=True)
    with open(args.features, 'r') as f:
        features = json.load(f)['features']

    # تحجيم البيانات
    train_scaled = train_df.copy()
    valid_scaled = valid_df.copy()
    for col in train_df.columns:
        s = MinMaxScaler()
        train_scaled[col] = s.fit_transform(train_df[[col]])
        valid_scaled[col] = s.transform(valid_df[[col]])

    # 3. إعداد البيانات للتدريب
    train_dataset = DatasetV1(train_scaled, target=args.target, features=features)
    valid_dataset = DatasetV1(valid_scaled, target=args.target, features=features)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], num_workers=0, shuffle=True)
    validation_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], num_workers=0, shuffle=False)

    # 4. تدريب النموذج
    early_stopping = EarlyStopping('val_loss', patience=10, verbose=False)

    model = LSTM(
        n_features=len(features),
        hidden_units=params['hidden_units'],
        n_layers=params['n_layers'],
        lr=params['learning_rate']
    )
    
    trainer = pl.Trainer(
        callbacks=[early_stopping],
        max_epochs=50,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        logger=False
    )

    try:
        trainer.fit(model, train_loader, validation_loader)
    except Exception as e:
        print(f"فشلت المحاولة بسبب خطأ: {e}")
        return float('inf')

    # 5. إرجاع النتيجة
    val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
    return val_loss

def main():
    parser = argparse.ArgumentParser(description='Tune hyperparameters for the LSTM model using Optuna.')
    parser.add_argument('--train', type=str, required=True, help='Path to the training data CSV.')
    parser.add_argument('--valid', type=str, required=True, help='Path to the validation data CSV.')
    parser.add_argument('--features', type=str, required=True, help='Path to the features JSON file.')
    parser.add_argument('--target', type=str, required=True, help='Target coin symbol (e.g., BTC).')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of optimization trials to run.')
    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print("\n--- اكتملت عملية التحسين ---")
    print(f"عدد المحاولات المكتملة: {len(study.trials)}")
    
    best_trial = study.best_trial
    print("\nأفضل محاولة:")
    print(f"  - قيمة الخسارة (Value): {best_trial.value:.6f}")
    
    print("  - أفضل المعلمات (Params):")
    for key, value in best_trial.params.items():
        print(f"    - {key}: {value}")

if __name__ == '__main__':
    main()