# api_client.py (النسخة النهائية والمعدلة)
import requests
import json
import pandas as pd
from datetime import datetime, timedelta

# استيراد الدوال من ملفاتك
from data_pull import fetch_crypto_data_from_coingecko
from feature_engineering import create_features 

# --- إعدادات العميل ---
BASE_API_URL = "http://localhost:8000" # تم تغيير الاسم إلى BASE
SEQUENCE_LENGTH = 60
DAYS_TO_FETCH = 200

# قائمة بكل العملات التي يحتاجها النموذج
COIN_LIST = [
    'btc', 'eth', 'usdt', 'usdc', 'bnb', 'xrp', 'busd', 'ada', 
    'sol', 'doge', 'dot', 'dai', 'shib', 'trx', 'avax', 'uni', 
    'wbtc', 'leo', 'ltc'
]

# --- دوال العميل ---
def prepare_payload(features_df: pd.DataFrame) -> dict:
    if len(features_df) < SEQUENCE_LENGTH:
        raise ValueError(f"البيانات غير كافية، نحتاج على الأقل {SEQUENCE_LENGTH} صفاً ولكن المتوفر {len(features_df)} صفاً فقط.")
    latest_sequence_df = features_df.tail(SEQUENCE_LENGTH).copy()
    if isinstance(latest_sequence_df.index, pd.DatetimeIndex):
        latest_sequence_df = latest_sequence_df.reset_index()
    for col in latest_sequence_df.columns:
        if pd.api.types.is_datetime64_any_dtype(latest_sequence_df[col]):
            latest_sequence_df[col] = latest_sequence_df[col].dt.strftime('%Y-%m-%d')
    sequence_as_list = latest_sequence_df.to_dict(orient='records')
    payload = {"sequence": sequence_as_list}
    return payload

def call_api(url: str, payload: dict) -> float:
    """تم تعديل الدالة لتستقبل رابط URL متغير."""
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result.get('prediction')
    except requests.exceptions.HTTPError as http_err:
        print(f"    !! خطأ في الـ API (HTTP Error): {http_err}")
        print(f"       تفاصيل الرد: {response.text}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"    !! خطأ في الاتصال بالـ API: {req_err}")
        return None

# === دالة main المعدلة بالكامل لتعالج كل عملة على حدة ===
def main():
    print("--- بدء عملية العميل (API Client) ---")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_FETCH)
    end_date_str = end_date.strftime('%d-%m-%Y')
    start_date_str = start_date.strftime('%d-%m-%Y')
    
    # حلقة تكرار لمعالجة كل عملة بشكل منفصل
    for coin in COIN_LIST:
        print(f"\n===== [ بدء المعالجة للعملة: {coin.upper()} ] =====")
        try:
            # الخطوة 1: جلب بيانات العملة الحالية فقط
            print(f"[ الخطوة 1/4 ] جلب بيانات {coin.upper()}...")
            raw_df = fetch_crypto_data_from_coingecko(coin, start_date_str, end_date_str)
            if raw_df is None:
                print(f"    - فشل جلب البيانات لـ {coin.upper()}. الانتقال للعملة التالية.")
                continue
            
            # الخطوة 2: حساب الميزات للعملة الحالية فقط
            print(f"[ الخطوة 2/4 ] حساب الميزات لـ {coin.upper()}...")
            raw_df.reset_index(inplace=True)
            raw_df['Coin'] = coin.upper() # إضافة عمود باسم العملة قد تحتاجه دالة create_features
            features_df = create_features(raw_df)
            print("    - تم حساب الميزات بنجاح.")

            # الخطوة 3: تجهيز الحمولة للعملة الحالية
            print(f"[ الخطوة 3/4 ] تجهيز حمولة JSON...")
            payload = prepare_payload(features_df)
            print("    - تم تجهيز الحمولة بنجاح.")

            # الخطوة 4: إرسال الطلب إلى الرابط الصحيح
            print(f"[ الخطوة 4/4 ] إرسال الطلب والحصول على التنبؤ...")
            
            # بناء الرابط الديناميكي الصحيح لكل عملة
            prediction_url = f"{BASE_API_URL}/predict/{coin}"
            
            prediction = call_api(prediction_url, payload)
            
            if prediction is not None:
                print(f"    ✅ نتيجة التنبؤ لـ {coin.upper()}: {prediction}")
            else:
                print(f"    ❌ فشلت عملية الحصول على التنبؤ لـ {coin.upper()}.")

        except Exception as e:
            print(f"!! حدث خطأ غير متوقع أثناء معالجة {coin.upper()}: {e}")

    print("\n--- انتهت العملية بالكامل ---")


if __name__ == "__main__":
    main()