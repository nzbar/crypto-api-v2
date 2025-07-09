# model_forecast.py (النسخة النهائية والمصححة)

import pandas as pd
import json
import torch
from pretrain.gru import GRU
from pretrain.lstm import LSTM
from sklearn.preprocessing import MinMaxScaler
import warnings

# تجاهل التحذيرات غير الهامة
warnings.filterwarnings("ignore")

# تحديد الجهاز (سيكون 'cpu' في بيئة Docker التي أنشأناها)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_prediction_assets(config_path, features_path, model_path, model_type, valid_data_path, target_coin):
    """
    تحميل جميع الأصول اللازمة للتنبؤ مرة واحدة عند بدء تشغيل الخادم.
    """
    print("Loading prediction assets...")
    
    # تحميل الإعدادات والميزات
    with open(config_path) as f: config = json.load(f)
    with open(features_path) as f: features = json.load(f)['features']
    
    # تحميل النموذج
    model_class = LSTM if model_type.lower() == 'lstm' else GRU
    model = model_class(
        n_features=len(features),
        hidden_units=config['hidden_units'],
        n_layers=config['n_layers'],
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # وضع النموذج في وضع التقييم (مهم جداً)
    
    # تحميل بيانات التحقق لتهيئة المحجمات (Scalers)
    valid_df = pd.read_csv(valid_data_path, index_col='Date', parse_dates=True)
    target_col_name = f"{target_coin.lower()}_avg_ohlc"
    
    # التأكد من وجود كل الأعمدة قبل تهيئة المحجمات
    all_required_cols = features + [target_col_name]
    if not all(col in valid_df.columns for col in all_required_cols):
        missing_cols = set(all_required_cols) - set(valid_df.columns)
        raise ValueError(f"ملف بيانات التحقق 'valid.csv' تنقصه الأعمدة التالية: {missing_cols}")

    # تهيئة المحجمات
    main_scaler = MinMaxScaler().fit(valid_df[all_required_cols])
    target_only_scaler = MinMaxScaler().fit(valid_df[[target_col_name]])
    
    print("Assets loaded successfully.")
    
    # إرجاع قاموس يحتوي على كل الأصول المحملة
    return {
        "model": model,
        "config": config,
        "features": features,
        "main_scaler": main_scaler,
        "target_only_scaler": target_only_scaler,
        "target_col_name": target_col_name
    }

def make_prediction(assets, input_data):
    """
    تقوم بعملية التنبؤ بناءً على بيانات التسلسل التي يتم إرسالها مباشرة في الطلب.
    """
    # نضع كل الكود في كتلة try واحدة لمعالجة أي خطأ بشكل آمن
    try:
        # --- 1. استخراج الأصول اللازمة ---
        model = assets['model']
        features = assets['features']
        main_scaler = assets['main_scaler']
        target_only_scaler = assets['target_only_scaler']
        target_col_name = assets['target_col_name']

        # --- 2. التحقق من هيكل الطلب الأساسي ---
        if 'sequence' not in input_data or not isinstance(input_data['sequence'], list) or not input_data['sequence']:
            raise ValueError("البيانات المرسلة يجب أن تكون كائن JSON يحتوي على مفتاح 'sequence' وقيمته قائمة غير فارغة.")

        input_sequence = input_data['sequence']

        # --- 3. تحويل البيانات المدخلة إلى DataFrame والتحقق من الميزات ---
        input_df = pd.DataFrame(input_sequence)
        
        required_features = set(features)
        provided_features = set(input_df.columns)
        
        if not required_features.issubset(provided_features):
            missing = sorted(list(required_features - provided_features))
            error_message = f"التسلسل المرسل تنقصه الميزات المطلوبة: {missing}"
            raise ValueError(error_message)

        # --- 4. تحجيم (Scale) بيانات الإدخال ---
        # نستخدم .copy() لتجنب التحذيرات، ونرتب الأعمدة لضمان تطابقها مع المحجم
        sequence_for_scaling = input_df[features].copy()
        sequence_for_scaling.loc[:, target_col_name] = 0
        
        # التأكد من أن ترتيب الأعمدة مطابق تماماً لما تم تدريب المحجم عليه
        ordered_columns = features + [target_col_name]
        sequence_for_scaling = sequence_for_scaling[ordered_columns]
        
        scaled_sequence = main_scaler.transform(sequence_for_scaling)[:, :len(features)]

        # --- 5. إجراء التنبؤ ---
        with torch.no_grad():
            input_tensor = torch.tensor(scaled_sequence, dtype=torch.float).unsqueeze(0).to(next(model.parameters()).device)
            y_hat = model(input_tensor)

        # --- 6. عكس التحجيم (Inverse Scale) ---
        prediction_scaled = y_hat.cpu().numpy().flatten()
        final_forecast = target_only_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()

        # --- 7. إرجاع النتيجة النهائية ---
        return float(final_forecast[0])

    except Exception as e:
        # التقاط أي خطأ يحدث في أي خطوة أعلاه وطباعته في سجل الخادم
        print(f"ERROR in make_prediction: An exception occurred: {e}")
        # نطلق الخطأ مرة أخرى ليتم التقاطه بواسطة معالج أخطاء Flask ويعيد رسالة 500
        raise e