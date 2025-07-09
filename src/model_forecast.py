# model_forecast.py (النسخة النهائية المصححة)

import pandas as pd
import json
import torch
from pretrain.gru import GRU
from pretrain.lstm import LSTM
from statsmodels.tsa.holtwinters import Holt
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_prediction_assets(config_path, features_path, model_path, model_type, valid_data_path, target_coin):
    """
    تحميل جميع الأصول اللازمة للتنبؤ مرة واحدة.
    """
    print("Loading prediction assets...")
    
    with open(config_path) as f: config = json.load(f)
    with open(features_path) as f: features = json.load(f)['features']
    
    model_class = LSTM if model_type.lower() == 'lstm' else GRU
    model = model_class(
        n_features=len(features),
        hidden_units=config['hidden_units'],
        n_layers=config['n_layers'],
        lr=config['learning_rate']
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    valid_df = pd.read_csv(valid_data_path, index_col='Date', parse_dates=True)
    target_col_name = f"{target_coin.lower()}_avg_ohlc"
    
    # --- التعديل الرئيسي هنا: فصل المحجمات ---
    # 1. محجم للميزات فقط (لتحويل مدخلات النموذج)
    features_scaler = MinMaxScaler().fit(valid_df[features])
    
    # 2. محجم للهدف فقط (لعكس تحويل مخرجات النموذج)
    target_scaler = MinMaxScaler().fit(valid_df[[target_col_name]])
    
    print("Assets loaded successfully.")
    
    return {
        "model": model,
        "config": config,
        "features": features,
        "features_scaler": features_scaler, # تم التغيير
        "target_scaler": target_scaler,     # تم التغيير
        "valid_df": valid_df,
        "target_col_name": target_col_name
    }

def make_prediction(assets, horizon=7):
    """
    تقوم بعملية التنبؤ باستخدام الأصول المحملة مسبقًا.
    """
    # استخراج الأصول من القاموس
    model = assets['model']
    valid_df = assets['valid_df']
    features = assets['features']
    config = assets['config']
    features_scaler = assets['features_scaler'] # تم التغيير
    target_scaler = assets['target_scaler']     # تم التغيير

    # 1. توقع الميزات المستقبلية باستخدام Holt
    pred_set = pd.DataFrame()
    for feature in features:
        holt = Holt(valid_df[feature], initialization_method="estimated").fit()
        pred = holt.forecast(horizon)
        pred_set = pd.concat([pred_set, pred], axis=1)
    pred_set.columns = features

    # 2. الحصول على آخر تسلسل من البيانات
    sequence_length = config.get('sequence_length', 60)
    last_sequence_unscaled = valid_df[features].iloc[-sequence_length:]
    
    # 3. دمج التسلسل الأخير مع الميزات المتوقعة
    full_sequence_unscaled = pd.concat([last_sequence_unscaled, pred_set]).iloc[-sequence_length:]
    
    # --- التعديل الرئيسي هنا: استخدام محجم الميزات فقط ---
    # 4. تحجيم (Scale) التسلسل بشكل صحيح
    full_sequence_scaled = features_scaler.transform(full_sequence_unscaled)
    
    # 5. إجراء التنبؤ
    with torch.no_grad():
        input_tensor = torch.tensor(full_sequence_scaled, dtype=torch.float).unsqueeze(0).to(DEVICE)
        y_hat = model(input_tensor)

    # 6. عكس التحجيم للحصول على القيمة الحقيقية
    prediction_scaled = y_hat.cpu().numpy().flatten()
    final_forecast = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()

    # طباعة النتيجة النهائية الصحيحة
    print(f"✅ Forecast complete. Prediction: {final_forecast[0]}")
    
    # 7. إرجاع النتيجة
    return float(final_forecast[0])