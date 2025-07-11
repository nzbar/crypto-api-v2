# app.py (النسخة النهائية - مع التحميل الكسول Lazy Loading)

import os
import glob
from datetime import datetime
from flask import Flask, request, jsonify, abort
import logging
import threading

# افترض أن دوالك موجودة في model_forecast.py
from model_forecast import load_prediction_assets, make_prediction

# --- إعداد التطبيق ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- إعدادات التحميل الكسول ---
# هذه هي المسارات داخل القرص الدائم المشترك على Render
MODELS_DIR = '/data/models'
DATA_DIR = '/data/data'

# قائمة بكل العملات التي يدعمها النظام
TARGET_COINS = [
    'btc', 'eth', 'usdt', 'usdc', 'bnb', 'xrp', 'busd', 'ada', 
    'sol', 'doge', 'dot', 'dai', 'shib', 'trx', 'avax', 'uni', 
    'wbtc', 'leo', 'ltc'
]

# سيبدأ القاموس فارغًا وسيتم ملؤه عند الطلب
assets_by_coin = {}
# قفل لضمان عدم تحميل نفس النموذج مرتين في نفس الوقت من عاملين مختلفين
assets_lock = threading.Lock()

# --- دوال مساعدة ---
def find_latest_file(directory, coin, prefix, suffix):
    """يبحث عن أحدث ملف بناءً على نمط معين."""
    search_pattern = os.path.join(directory, f"{prefix}{coin}_*{suffix}")
    files = glob.glob(search_pattern)
    if not files: return None
    
    # يختار الملف الأحدث بناءً على تاريخ تعديل الملف
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def load_single_asset(coin):
    """
    دالة جديدة تقوم بتحميل الأصول لعملة واحدة فقط عند الحاجة.
    """
    app.logger.info(f"--- [LAZY LOADING] Attempting to load assets for {coin.upper()} ---")
    try:
        MODEL_TYPE = 'lstm'
        # هذه الملفات يتم نسخها مع الصورة وليست على القرص الدائم
        CONFIG_PATH = 'config/config_nn.json'
        FEATURES_PATH = 'config/features.json'
        
        MODEL_PATH = find_latest_file(MODELS_DIR, coin, f"{MODEL_TYPE}_", ".pth")
        VALID_DATA_PATH = find_latest_file(DATA_DIR, coin, "", ".csv")

        if not MODEL_PATH or not VALID_DATA_PATH:
            raise FileNotFoundError(f"Could not find model or data files for {coin.upper()} in persistent storage ({MODELS_DIR}, {DATA_DIR})")

        coin_assets = load_prediction_assets(
            CONFIG_PATH, FEATURES_PATH, MODEL_PATH, MODEL_TYPE, VALID_DATA_PATH, coin
        )
        app.logger.info(f"--- [LAZY LOADING] Assets for {coin.upper()} loaded and cached successfully.")
        return coin_assets

    except Exception as e:
        app.logger.error(f"--- [LAZY LOADING] FATAL: Could not load assets for {coin.upper()}. Error: {e}")
        return None

# --- نقاط النهاية (Endpoints) ---

@app.route('/health', methods=['GET'])
def health_check():
    """الآن فحص الصحة يظهر فقط النماذج التي تم تحميلها فعلاً حتى الآن."""
    return jsonify({
        "status": "ok",
        "supported_coins": TARGET_COINS,
        "currently_loaded_models": list(assets_by_coin.keys())
    }), 200

@app.route('/predict/<string:coin>', methods=['POST'])
def handle_prediction(coin):
    coin = coin.lower()
    if coin not in TARGET_COINS:
        abort(404, description=f"Coin '{coin}' is not supported.")

    # --- منطق التحميل الكسول ---
    if coin not in assets_by_coin:
        # استخدام قفل لضمان عدم حدوث تضارب إذا جاء طلبان لنفس العملة في نفس اللحظة
        with assets_lock:
            # تحقق مرة أخرى بعد الحصول على القفل، فقد يكون عامل آخر قد قام بالتحميل
            if coin not in assets_by_coin:
                new_assets = load_single_asset(coin)
                if new_assets is None:
                    abort(503, description=f"Model for coin '{coin}' is temporarily unavailable or failed to load.")
                assets_by_coin[coin] = new_assets

    # الآن الأصول موجودة بالتأكيد في الذاكرة
    current_assets = assets_by_coin[coin]
    
    if not request.is_json:
        abort(400, description="Invalid request format. Expecting a JSON body.")

    data = request.get_json()
    
    try:
        prediction = make_prediction(assets=current_assets, input_data=data)
        return jsonify({"coin": coin, "prediction": prediction})

    except Exception as e:
        app.logger.error(f"An unexpected error occurred during prediction for {coin.upper()}: {e}")
        abort(500)

# --- معالجات الأخطاء ---
@app.errorhandler(400)
def bad_request(error): return jsonify({"error": "Bad Request", "message": error.description or "Invalid data received."}), 400
@app.errorhandler(404)
def not_found(error): return jsonify({"error": "Not Found", "message": error.description or "This resource does not exist."}), 404
@app.errorhandler(500)
def internal_server_error(error): return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred on our end."}), 500
@app.errorhandler(503)
def service_unavailable(error): return jsonify({"error": "Service Unavailable", "message": error.description or "The service is not ready to handle requests."}), 503

# --- تشغيل التطبيق (للإختبار المحلي فقط) ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)