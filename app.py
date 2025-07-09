# app.py (النسخة النهائية والمعدلة)

import os
import glob
from datetime import datetime
from flask import Flask, request, jsonify, abort
import logging

# افترض أن دوالك موجودة في model_forecast.py
from model_forecast import load_prediction_assets, make_prediction

# --- إعداد التطبيق ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


# --- تعريف المسارات المطلقة بناءً على إعدادات render.yaml ---
# هذا يضمن أن التطبيق يقرأ ويكتب من القرص الصلب الدائم
MODELS_DIR = '/data/models'
DATA_DIR = '/data/data' # افترضنا أن ملفات csv ستكون في مجلد 'data' داخل القرص


# --- دوال مساعدة ووظائف تحميل النماذج ---

def find_latest_file(directory, coin, prefix, suffix):
    """
    تبحث هذه الدالة في مجلد معين عن أحدث ملف يطابق نمطًا محددًا.
    """
    search_pattern = os.path.join(directory, f"{prefix}{coin}_*{suffix}")
    files = glob.glob(search_pattern)
    if not files:
        return None

    latest_file = None
    latest_date = None

    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            date_str = filename.replace(f"{prefix}{coin}_", "").replace(suffix, "")
            current_date = datetime.strptime(date_str, '%d%m%Y')
            if latest_date is None or current_date > latest_date:
                latest_date = current_date
                latest_file = file_path
        except ValueError:
            app.logger.warning(f"Ignoring file with incorrect date format: {filename}")
            continue
    return latest_file

def load_all_assets(target_coins):
    """
    يقوم بتحميل أصول النماذج لجميع العملات المستهدفة.
    """
    loaded_assets = {}
    for coin in target_coins:
        app.logger.info(f"--- Loading assets for {coin.upper()} ---")
        try:
            MODEL_TYPE = 'lstm'
            CONFIG_PATH = 'config/config_nn.json'
            FEATURES_PATH = 'config/features.json'
            
            # البحث عن أحدث الملفات ديناميكياً في المسارات المطلقة
            MODEL_PATH = find_latest_file(MODELS_DIR, coin, f"{MODEL_TYPE}_", ".pth")
            VALID_DATA_PATH = find_latest_file(DATA_DIR, coin, "", ".csv")

            if not MODEL_PATH or not VALID_DATA_PATH:
                raise FileNotFoundError(f"Could not find model or data files for {coin.upper()} in persistent storage")

            # تحميل الأصول للعملة الحالية
            coin_assets = load_prediction_assets(
                CONFIG_PATH, FEATURES_PATH, MODEL_PATH, MODEL_TYPE, VALID_DATA_PATH, coin
            )
            loaded_assets[coin] = coin_assets
            app.logger.info(f"Assets for {coin.upper()} loaded successfully using {os.path.basename(MODEL_PATH)}")

        except Exception as e:
            app.logger.error(f"FATAL: Could not load assets for {coin.upper()}. Error: {e}")
    return loaded_assets

# --- تحميل أصول جميع النماذج عند بدء التشغيل ---
# تم تحديث القائمة لتشمل كل العملات من ملف العميل
TARGET_COINS = [
    'btc', 'eth', 'usdt', 'usdc', 'bnb', 'xrp', 'busd', 'ada', 
    'sol', 'doge', 'dot', 'dai', 'shib', 'trx', 'avax', 'uni', 
    'wbtc', 'leo', 'ltc'
]
assets_by_coin = load_all_assets(TARGET_COINS)


# --- نقاط النهاية (Endpoints) ---

@app.route('/health', methods=['GET'])
def health_check():
    """نقطة نهاية للتحقق من صحة الخدمة والنماذج التي تم تحميلها."""
    loaded_successfully = list(assets_by_coin.keys())
    failed_to_load = [coin for coin in TARGET_COINS if coin not in loaded_successfully]
    
    status_code = 200 if len(failed_to_load) == 0 else 503
    
    return jsonify({
        "status": "ok" if status_code == 200 else "unhealthy",
        "supported_coins": TARGET_COINS,
        "loaded_models": loaded_successfully,
        "failed_models": failed_to_load
    }), status_code

@app.route('/info/<string:coin>', methods=['GET'])
def model_info(coin):
    """إرجاع معلومات عن النموذج المستخدم حالياً لعملة معينة."""
    coin = coin.lower()
    if coin in assets_by_coin and 'model_info' in assets_by_coin[coin]:
        return jsonify(assets_by_coin[coin]['model_info']), 200
    else:
        abort(404, description=f"Information not available for coin '{coin}'. It might not be supported or failed to load.")

@app.route('/predict/<string:coin>', methods=['POST'])
def handle_prediction(coin):
    """نقطة النهاية الرئيسية لعمل التنبؤ لعملة معينة."""
    coin = coin.lower()
    if coin not in assets_by_coin:
        abort(404, description=f"Prediction service is not available for '{coin}'. Model not found.")
        
    if not request.is_json:
        abort(400, description="Invalid request format. Expecting a JSON body.")

    data = request.get_json()
    
    try:
        prediction = make_prediction(assets=assets_by_coin[coin], input_data=data)
        return jsonify({"coin": coin, "prediction": prediction})

    except KeyError as e:
        abort(400, description=f"Missing required field in request: {e}")
    except Exception as e:
        app.logger.error(f"An unexpected error occurred during prediction for {coin.upper()}: {e}")
        abort(500)


# --- معالجات الأخطاء (Error Handlers) ---
@app.errorhandler(400)
def bad_request(error): return jsonify({"error": "Bad Request", "message": error.description or "Invalid data received."}), 400
@app.errorhandler(404)
def not_found(error): return jsonify({"error": "Not Found", "message": error.description or "This resource does not exist."}), 404
@app.errorhandler(500)
def internal_server_error(error): return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred on our end."}), 500
@app.errorhandler(503)
def service_unavailable(error): return jsonify({"error": "Service Unavailable", "message": error.description or "The service is not ready to handle requests."}), 503


# --- تشغيل التطبيق ---
if __name__ == '__main__':
    # استخدم المنفذ 8000 ليتوافق مع Dockerfile
    app.run(debug=True, port=8000)