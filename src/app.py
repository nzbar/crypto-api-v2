# app.py

from flask import Flask, request, jsonify
from model_forecast import load_prediction_assets, make_prediction
import os

# --- الإعدادات الأساسية ---
# هذه هي نفس المسارات التي كانت في argparse
# يجب أن تكون هذه الملفات موجودة على الخادم
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.json')
FEATURES_PATH = os.path.join(BASE_DIR, 'config', 'features.json')
MODEL_PATH = os.path.join(BASE_DIR, 'pretrained_models', 'gru_model.pth')
VALID_DATA_PATH = os.path.join(BASE_DIR, 'data', 'validation.csv')
TARGET_COIN = 'bitcoin' # العملة المستهدفة
MODEL_TYPE = 'gru'      # نوع النموذج

# --- تهيئة التطبيق وتحميل الأصول ---
app = Flask(__name__)

# تحميل الأصول مرة واحدة عند بدء تشغيل الخادم وتخزينها في متغير
try:
    prediction_assets = load_prediction_assets(
        config_path=CONFIG_PATH,
        features_path=FEATURES_PATH,
        model_path=MODEL_PATH,
        model_type=MODEL_TYPE,
        valid_data_path=VALID_DATA_PATH,
        target_coin=TARGET_COIN
    )
except FileNotFoundError as e:
    print(f"Error loading assets: {e}. Make sure all paths are correct.")
    prediction_assets = None # Handle case where loading fails

# --- تحديد رابط الـ API ---
@app.route('/predict', methods=['GET'])
def handle_prediction():
    if prediction_assets is None:
        return jsonify({'error': 'Prediction assets could not be loaded. Check server logs.'}), 500

    try:
        # يمكن استقبال متغيرات من الطلب إذا أردت، مثل عدد أيام التنبؤ
        # horizon = request.args.get('horizon', default=7, type=int)
        
        # استدعاء دالة التنبؤ السريعة
        forecasted_price = make_prediction(prediction_assets) #, horizon=horizon)
        
        # إرجاع النتيجة على هيئة JSON
        return jsonify({
            'status': 'success',
            'predicted_price': f'{forecasted_price:.2f}'
        })

    except Exception as e:
        # التعامل مع أي أخطاء تحدث أثناء عملية التنبؤ
        app.logger.error(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An internal error occurred during prediction.'}), 500

# --- تشغيل التطبيق ---
if __name__ == '__main__':
    # عند استخدام خادم إنتاج مثل Gunicorn، لن يتم تنفيذ هذا السطر
    # Gunicorn سيقوم بتشغيل التطبيق مباشرة
    app.run(host='0.0.0.0', port=5000, debug=False)