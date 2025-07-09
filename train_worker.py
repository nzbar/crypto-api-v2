# train_worker.py (النسخة النهائية والمعدلة)

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime, timedelta
import os
import numpy as np

# استيراد دوالك ونماذجك
from data_pull import fetch_crypto_data_from_coingecko
from feature_engineering import create_features
from pretrain.lstm import LSTM

# --- الإعدادات ---
# المسارات إلى القرص الصلب الدائم في Render
MODELS_OUTPUT_DIR = '/data/models' 
DATA_OUTPUT_DIR = '/data/data' # مجلد لحفظ ملفات البيانات المستخدمة للتدريب
LOGS_DIR = '/data/logs' # مجلد لحفظ سجلات التدريب

COIN_LIST = [
    'btc', 'eth', 'usdt', 'usdc', 'bnb', 'xrp', 'busd', 'ada', 
    'sol', 'doge', 'dot', 'dai', 'shib', 'trx', 'avax', 'uni', 
    'wbtc', 'leo', 'ltc'
]
DAYS_TO_FETCH = 365
SEQUENCE_LENGTH = 60
BATCH_SIZE = 64
MAX_EPOCHS = 50

# --- دوال مساعدة ---

def create_sequences(input_data: pd.DataFrame, target_column: str, sequence_length: int):
    """
    يقوم بتحويل DataFrame إلى تسلسلات مناسبة لنماذج LSTM/GRU.
    """
    sequences = []
    labels = []
    features = [col for col in input_data.columns if col != target_column]
    
    data_values = input_data[features].values
    label_values = input_data[target_column].values

    for i in range(len(input_data) - sequence_length):
        sequences.append(data_values[i:i + sequence_length])
        labels.append(label_values[i + sequence_length])
        
    return torch.tensor(np.array(sequences), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.float32)

def prepare_dataloaders(features_df, target_col_name, sequence_length, batch_size):
    """
    تجهيز محملات البيانات للتدريب والتحقق باستخدام التسلسلات.
    """
    train_size = int(len(features_df) * 0.9)
    train_df, val_df = features_df.iloc[:train_size], features_df.iloc[train_size:]

    X_train, y_train = create_sequences(train_df, target_col_name, sequence_length)
    X_val, y_val = create_sequences(val_df, target_col_name, sequence_length)
    
    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        return None, None, -1 # لا توجد بيانات كافية

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    n_features = X_train.shape[2]
    return train_loader, val_loader, n_features


def run_training_job():
    print("--- [WORKER] بدء مهمة التدريب المجدولة ---")

    # إنشاء المجلدات الرئيسية إذا لم تكن موجودة
    os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # --- 1. جلب البيانات وهندسة الميزات (مرة واحدة لكل العملات) ---
    print("[1/3] جلب ومعالجة البيانات لجميع العملات...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_FETCH)
    all_raw_dfs = []
    for coin in COIN_LIST:
        coin_df = fetch_crypto_data_from_coingecko(coin, start_date.strftime('%d-%m-%Y'), end_date.strftime('%d-%m-%Y'))
        if coin_df is not None:
            coin_df['Coin'] = coin.upper()
            all_raw_dfs.append(coin_df)

    if not all_raw_dfs:
        print("❌ فشل جلب أي بيانات، إيقاف المهمة.")
        return

    raw_df = pd.concat(all_raw_dfs)
    raw_df.reset_index(inplace=True)
    features_df = create_features(raw_df)
    print("  - تم تجهيز بيانات الميزات بنجاح.")

    # --- 2. حلقة تدريب لكل عملة على حدة ---
    for coin in COIN_LIST:
        print(f"\n===== [ بدء تدريب النموذج لعملة: {coin.upper()} ] =====")
        try:
            # تعريف العمود المستهدف ديناميكياً
            target_col = f"{coin.lower()}_avg_ohlc"
            if target_col not in features_df.columns:
                print(f"  - العمود المستهدف '{target_col}' غير موجود، تخطي هذه العملة.")
                continue

            # تجهيز محملات البيانات لهذه العملة
            train_loader, val_loader, n_features = prepare_dataloaders(features_df, target_col, SEQUENCE_LENGTH, BATCH_SIZE)
            
            if train_loader is None:
                print(f"  - لا توجد بيانات كافية لتدريب نموذج {coin.upper()}.")
                continue

            # إعداد نقاط الحفظ والتوقف المبكر
            # ** تعديل جذري هنا لحفظ الملف بالاسم الصحيح الذي يفهمه app.py **
            current_date_str = datetime.now().strftime("%d%m%Y")
            checkpoint_callback = ModelCheckpoint(
                dirpath=MODELS_OUTPUT_DIR,
                # حفظ الملف بالاسم واللاحقة الصحيحين
                filename=f'lstm_{coin}_{current_date_str}',
                save_top_k=1,
                verbose=True,
                monitor='val_loss',
                mode='min'
            )
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=True)

            # تهيئة النموذج
            model = LSTM(n_features=n_features, lr=1e-4)

            # تهيئة المدرب
            trainer = pl.Trainer(
                max_epochs=MAX_EPOCHS,
                accelerator='cpu',
                callbacks=[checkpoint_callback, early_stopping_callback],
                logger=pl.loggers.CSVLogger(save_dir=LOGS_DIR, name=f'{coin}_training_logs'),
                enable_progress_bar=False # مناسب للتشغيل في الخلفية
            )

            # بدء التدريب
            print(f"  - بدء التدريب الفعلي لنموذج {coin.upper()}...")
            trainer.fit(model, train_loader, val_loader)

            if checkpoint_callback.best_model_path:
                print(f"  - ✅ اكتمل تدريب {coin.upper()}! تم حفظ أفضل نموذج في: {checkpoint_callback.best_model_path}")
                # حفظ نسخة من البيانات المستخدمة مع تاريخ اليوم لضمان التوافق
                data_filename = f"{coin}_{current_date_str}.csv"
                data_save_path = os.path.join(DATA_OUTPUT_DIR, data_filename)
                features_df.to_csv(data_save_path, index=False)
                print(f"  - تم حفظ نسخة البيانات المستخدمة في: {data_save_path}")
            else:
                 print(f"  - ❌ فشل تدريب {coin.upper()} أو لم يتم تحقيق تحسن لحفظ النموذج.")

        except Exception as e:
            print(f"  - ‼️ حدث خطأ فادح أثناء تدريب نموذج {coin.upper()}: {e}")

    print("\n--- ✅ نجحت مهمة التدريب المجدولة لجميع العملات ---")

if __name__ == "__main__":
    run_training_job()