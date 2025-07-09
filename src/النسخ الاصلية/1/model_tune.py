"""
File: model_tune.py
Description: تدريب نموذج LSTM المطور وحفظه مع محول القياس في BentoML.
File Created: 07/06/2025
Python Version: 3.9
"""
import os, argparse, sys, json, warnings
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pretrain.lstm_tuned import LSTMTuned 
from pretrain.datasets import DatasetV1
#from pretrain.datasets import Dataset  # بدلاً من DatasetV1
import bentoml

warnings.filterwarnings("ignore")

def parse_arguments():
    """تحليل وسيطات سطر الأوامر."""
    parser = argparse.ArgumentParser(description='تدريب النموذج المطور.')
    parser.add_argument('-tr', '--train', required=True, help='مسار ملف بيانات التدريب.')
    parser.add_argument('-vd', '--valid', required=True, help='مسار ملف بيانات التحقق.')
    parser.add_argument('-t', '--target', required=True, help='العملة المستهدفة (e.g., BTC).')
    parser.add_argument('-ft', '--features', required=True, help='مسار ملف الميزات JSON.')
    parser.add_argument('-c', '--config', required=True, help='مسار ملف الإعدادات JSON.')
    return parser.parse_args()

def load_and_preprocess_data(train_path, valid_path, features_path, target_symbol):
    """تحميل البيانات ومعالجتها."""
    print("جاري تحميل ومعالجة البيانات...")
    train_df = pd.read_csv(train_path, index_col='Date', parse_dates=True)
    valid_df = pd.read_csv(valid_path, index_col='Date', parse_dates=True)
    with open(features_path, 'r') as f:
        features = json.load(f)['features']

    # --- التصحيح: بناء اسم العمود المستهدف بشكل صحيح ---
    target_col_name = target_symbol.upper()
    
    # التأكد من أن العمود المستهدف ليس ضمن الميزات
    if target_col_name in features:
        print(f"تحذير: العمود المستهدف '{target_col_name}' موجود في قائمة الميزات. سيتم إزالته.")
        features.remove(target_col_name)

    target_and_features = features + [target_col_name]
    
    # التأكد من وجود جميع الأعمدة قبل المتابعة
    missing_cols = [col for col in target_and_features if col not in train_df.columns]
    if missing_cols:
        raise KeyError(f"الأعمدة التالية غير موجودة في البيانات: {missing_cols}")

    train_subset = train_df[target_and_features]
    valid_subset = valid_df[target_and_features]

    scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_subset), index=train_subset.index, columns=train_subset.columns)
    valid_scaled = pd.DataFrame(scaler.transform(valid_subset), index=valid_subset.index, columns=valid_subset.columns)
    
    print("اكتملت معالجة البيانات.")
    return train_scaled, valid_scaled, features, scaler, target_col_name

def tune_and_train_model(config, train_data, valid_data, features, target_col_name):
    """تدريب النموذج المطور."""
    print(f"بدء تدريب النموذج المطور...")
    pl.seed_everything(config.get('seed', 42))
    
    # --- ملاحظة: يجب التأكد من أن DatasetV1 أو Dataset تقبل البيانات بهذا الشكل ---
    train_dataset = DatasetV1(train_data, target=target_col_name, features=features)
    valid_dataset = DatasetV1(valid_data, target=target_col_name, features=features)
    
    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), num_workers=config.get('num_workers', 0), shuffle=True, persistent_workers=True if config.get('num_workers', 0) > 0 else False)
    validation_loader = DataLoader(valid_dataset, batch_size=config.get('batch_size', 32), num_workers=config.get('num_workers', 0), persistent_workers=True if config.get('num_workers', 0) > 0 else False)
    
    early_stopping = EarlyStopping('val_loss', patience=config.get('patience', 15), verbose=True, min_delta=0.0001)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, verbose=True)

    model = LSTMTuned(
        n_features=len(features),
        hidden_units=config.get('hidden_units', 128),
        n_layers=config.get('n_layers', 2),
        lr=config.get('learning_rate', 0.0005),
        dropout_prob=config.get('dropout_prob', 0.2)
    )
    
    # --- التصحيح: استخدام إعدادات المدرب الحديثة من ملف الإعدادات ---
    trainer = pl.Trainer(
        callbacks=[early_stopping, checkpoint_callback],
        max_epochs=config.get('n_epochs', 100),
        accelerator=config['accelerator'], 
        devices=config['devices'],
        gradient_clip_val=1.0
    )
    
    trainer.fit(model, train_loader, validation_loader)
    print("اكتمل التدريب.")
    
    best_model_path = checkpoint_callback.best_model_path
    print(f"تحميل أفضل نموذج من المسار: {best_model_path}")
    best_model = LSTMTuned.load_from_checkpoint(best_model_path)
    return best_model

def save_to_bentoml(model_object, scaler_object, model_name):
    """حفظ النموذج والمحول معًا في BentoML."""
    print(f"جاري حفظ النموذج المطور والمحول '{model_name}' في متجر BentoML...")
    saved_model = bentoml.pytorch.save_model(
        model_name, 
        model_object, 
        custom_objects={"scaler": scaler_object}
    )
    print(f"تم الحفظ بنجاح. العلامة (Tag): {saved_model.tag}")

def main():
    args = parse_arguments()
    print(f"\n--- بدء عملية تدريب وتحسين النموذج لـ {args.target} ---")
    
    try:
        with open(args.config, 'r') as f: config = json.load(f)

        train_scaled, valid_scaled, features, scaler, target_col_name = load_and_preprocess_data(
            args.train, args.valid, args.features, args.target
        )

        tuned_model = tune_and_train_model(config, train_scaled, valid_scaled, features, target_col_name)
        
        if tuned_model:
            bento_model_name = f"lstm-tuned_{args.target.lower()}"
            save_to_bentoml(tuned_model, scaler, bento_model_name)
        
        print("\n--- انتهت عملية تحسين النموذج بنجاح ---\n")
    except (KeyError, FileNotFoundError) as e:
        print(f"\nخطأ فادح: {e}")
        print("يرجى التأكد من أن ملف الميزات 'features.json' يحتوي على أسماء أعمدة صحيحة وموجودة في ملف البيانات.")
        sys.exit(1)


if __name__ == '__main__':
    main()