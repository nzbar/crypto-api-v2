"""
File: data_pull.py
Description: Dataset generator (source: CoinMarketCap Pro API or CoinGecko API).
File Created: 06/09/2023 (Modified for CoinGecko & Source Selection: 30/05/2025)
Python Version: 3.9

Usage:
  data_pull.py [--path=<path>] [--filename=<filename>] [--coins=<coins>] [--start=<start_date>] [--end=<end_date>] [--source=<source_api>]
  data_pull.py -h | --help

Options:
  -h --help                         إظهار شاشة المساعدة هذه.
  --path=<path>                     مسار مجلد الإخراج [القيمة الافتراضية: .].
  --filename=<filename>             اسم ملف الإخراج [القيمة الافتراضية: dataset.csv].
  --coins=<coins>                   قائمة رموز العملات المشفرة مفصولة بفاصلة (مثال: btc,eth,ada) [القيمة الافتراضية: btc,eth,usdt,usdc,bnb,xrp,busd,ada,sol,doge,matic,dot,dai,shib,trx,avax,uni,wbtc,leo,ltc].
  --start=<start_date>              تاريخ البدء (DD-MM-YYYY) [القيمة الافتراضية: 31-05-2025].
  --end=<end_date>                  تاريخ الانتهاء (DD-MM-YYYY) [القيمة الافتراضية: 01-06-2025].
  --source=<source_api>             مصدر API (CoinMarketCap أو CoinGecko) [القيمة الافتراضية: CoinGecko].
"""

# Imports
import os
import json
import argparse
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
import time
from docopt import docopt # <--- تأكد من وجود هذا السطر وغير معلق
# --- دالة CoinMarketCap الأصلية (لم يتم تعديلها) ---
def fetch_crypto_ohlc_data_from_api(crypto_symbol, start_date_str, end_date_str):
    """
    تجلب بيانات سعر OHLC لعملة مشفرة معينة من CoinMarketCap Pro API.

    الوسائط (Args):
        crypto_symbol (str): رمز العملة المشفرة (مثل 'btc', 'eth').
        start_date_str (str): تاريخ البدء بتنسيق 'DD-MM-YYYY'.
        end_date_str (str): تاريخ الانتهاء بتنسيق 'DD-MM-YYYY'.

    الإرجاع (Returns):
        pd.DataFrame: إطار بيانات (DataFrame) يحتوي على أعمدة 'Date', 'Open', 'High', 'Low', 'Close', 'Coin',
                      أو None إذا تعذر جلب البيانات.
    """
    print(f"جاري محاولة جلب البيانات لـ {crypto_symbol.upper()} من {start_date_str} إلى {end_date_str} باستخدام CoinMarketCap Pro API...")

    # --- هام: استبدل هذا بمفتاح API الفعلي الخاص بك من CoinMarketCap Pro ---
    # ملاحظة: نقطة نهاية API لـ CoinMarketCap للحصول على بيانات OHLCV التاريخية (v1/cryptocurrency/ohlcv/historical)
    # عادة ما تكون متاحة فقط في الخطط المدفوعة (مثل Professional, Enterprise).
    # إذا كنت تستخدم مفتاح API مجاني/أساسي، فمن المحتمل أن ترجع نقطة النهاية هذه خطأ.
    CMC_API_KEY = "b89b7362-8d63-484c-82a3-204be607b637"
    # يمكنك أيضًا اختيار تخزين هذا المفتاح في متغير بيئة لمزيد من الأمان:
    # CMC_API_KEY = os.getenv("CMC_PRO_API_KEY")

    BASE_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"

    headers = {
        'X-CMC_PRO_API_KEY': CMC_API_KEY,
        'Accepts': 'application/json'
    }

    try:
        start_dt_obj = datetime.strptime(start_date_str, '%d-%m-%Y')
        end_dt_obj = datetime.strptime(end_date_str, '%d-%m-%Y')

        # واجهة برمجة تطبيقات CoinMarketCap تتوقع تنسيق ISO 8601 للطوابع الزمنية
        # مثال: 2024-05-24T00:00:00Z
        time_start = start_dt_obj.isoformat(timespec='seconds') + 'Z'
        time_end = end_dt_obj.isoformat(timespec='seconds') + 'Z'

        params = {
            'symbol': crypto_symbol.upper(), # الرمز (مثل BTC, ETH)
            'time_start': time_start,
            'time_end': time_end,
            'interval': 'daily',             # طلب بيانات يومية
            'convert': 'USD'                 # تحويل الأسعار إلى الدولار الأمريكي
        }

        # أضف تأخيرًا صغيرًا لتكون مهذبًا مع الـ API، على الرغم من أن حدود CMC عادةً ما تكون سخية مع مفتاح Pro
        time.sleep(8)

        response = requests.get(BASE_URL, headers=headers, params=params)
        response.raise_for_status() # أرفع خطأ HTTP للاستجابات السيئة (4xx أو 5xx)

        json_data = response.json()

        # التحقق من أخطاء الـ API المبلغ عنها بواسطة CoinMarketCap في حقل 'status'
        if json_data.get('status', {}).get('error_code', 0) != 0:
            error_message = json_data.get('status', {}).get('error_message', 'خطأ غير معروف في API.')
            print(f"خطأ في CoinMarketCap API لـ {crypto_symbol.upper()}: {error_message}")
            if "This API endpoint is only available to customers on a paid plan" in error_message:
                print("ملاحظة: البيانات التاريخية لـ OHLCV متاحة عادةً فقط في الخطط المدفوعة لـ CoinMarketCap.")
                print("إذا كنت تستخدم مفتاح API مجاني/للمطورين، فلن تعمل نقطة النهاية هذه.")
            return None

        # الوصول إلى البيانات بناءً على هيكل استجابة الـ API: data -> {SYMBOL} -> quotes
        coin_data = json_data.get('data', {}).get(crypto_symbol.upper())
        if not coin_data:
            print(f"لم يتم العثور على بيانات لـ {crypto_symbol.upper()} في استجابة CoinMarketCap API. "
                    f"قد يكون رمزًا غير صالح أو لا توجد بيانات للنطاق الزمني المحدد.")
            return None

        ohlcv_quotes = coin_data.get('quotes', [])

        if not ohlcv_quotes:
            print(f"لم يتم تلقي اقتباسات OHLCV من CoinMarketCap لـ {crypto_symbol.upper()} للفترة المحددة.")
            return None

        # استخراج بيانات OHLC وتنسيقها
        extracted_data = []
        for quote in ohlcv_quotes:
            # 'time_open' هو الطابع الزمني ISO 8601 لبداية شمعة اليومية
            time_open_str = quote.get('time_open')
            ohlc_usd = quote.get('quote', {}).get('USD', {}) # بيانات OHLC متداخلة تحت 'quote' و 'USD'

            # تحويل سلسلة ISO 8601 إلى تنسيق 'DD-MM-YYYY'
            date = datetime.fromisoformat(time_open_str.replace('Z', '+00:00')).strftime('%d-%m-%Y')
            open_price = ohlc_usd.get('open')
            high_price = ohlc_usd.get('high')
            low_price = ohlc_usd.get('low')
            close_price = ohlc_usd.get('close')
            volume_price = ohlc_usd.get('volume') # إضافة جلب الحجم

            # التأكد من وجود جميع قيم OHLC المطلوبة قبل الإضافة إلى القائمة
            if all(v is not None for v in [date, open_price, high_price, low_price, close_price, volume_price]):
                extracted_data.append({
                    'Date': date,
                    'Open': open_price,
                    'High': high_price,
                    'Low': low_price,
                    'Close': close_price,
                    'Volume': volume_price, # إضافة الحجم
                    'Coin': crypto_symbol.upper()
                })

        if not extracted_data:
            print(f"لم يتم استخراج بيانات OHLCV صالحة لـ {crypto_symbol.upper()} بعد تحليل استجابة الـ API.")
            return None

        df_final = pd.DataFrame(extracted_data)

        # تصفية نهائية للالتزام بدقة بنطاق التاريخ المطلوب،
        # في حال أرجعت الـ API بيانات أكثر قليلاً بسبب تعريفات المنطقة الزمنية/حدود اليوم.
        df_final['Date_dt'] = pd.to_datetime(df_final['Date'], format='%d-%m-%Y')
        df_filtered = df_final[(df_final['Date_dt'] >= start_dt_obj) & (df_final['Date_dt'] <= end_dt_obj)]
        df_filtered = df_filtered.drop(columns=['Date_dt']) # إسقاط عمود التاريخ المؤقت

        if df_filtered.empty:
            print(f"لم يتم العثور على بيانات لـ {crypto_symbol.upper()} ضمن نطاق التاريخ المطلوب {start_date_str} إلى {end_date_str} بعد التصفية النهائية.")
            return None

        return df_filtered

    except requests.exceptions.RequestException as e:
        print(f"خطأ في جلب البيانات لـ {crypto_symbol.upper()} من CoinMarketCap Pro API: {e}")
        # اطبع محتوى الاستجابة لتصحيح أخطاء HTTP إن أمكن
        if hasattr(e, 'response') and e.response is not None:
            print(f"محتوى الاستجابة: {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"خطأ في تحليل استجابة JSON من CoinMarketCap لـ {crypto_symbol.upper()}: {e}")
        # حاول طباعة نص الاستجابة إذا كان متاحًا لتصحيح الأخطاء
        if 'response' in locals() and response is not None:
            print(f"المحتوى المستلم (أول 500 حرف): {response.text[:500]}")
        return None
    except KeyError as e:
        print(f"خطأ في هيكل البيانات لـ {crypto_symbol.upper()} من CoinMarketCap: مفتاح مفقود {e}. قد تكون الاستجابة مشوهة.")
        if 'response' in locals() and response is not None:
            print(f"المحتوى المستلم (أول 500 حرف): {response.text[:500]}")
        return None
    except Exception as e:
        print(f"حدث خطأ غير متوقع لـ {crypto_symbol.upper()}: {e}")
        return None

# --- نهاية دالة CoinMarketCap الأصلية ---

## إضافة دعم CoinGecko (دالة جديدة)

#```python
# --- إعدادات CoinGecko API ---
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# قاموس لتحويل رموز العملات الشائعة إلى معرفات CoinGecko (IDs)
# هذا القاموس ضروري لأن CoinGecko يستخدم IDs مثل 'bitcoin' بدلاً من الرموز مثل 'btc'
# يرجى توسيع هذه القائمة إذا كنت بحاجة إلى عملات إضافية غير موجودة
COINGECKO_IDS = {
    'btc': 'bitcoin',
    'eth': 'ethereum',
    'usdt': 'tether',
    'usdc': 'usd-coin',
    'bnb': 'binancecoin',
    'xrp': 'ripple',
    'busd': 'binance-usd', # ملاحظة: BUSD تم إيقافها، قد تكون البيانات نادرة أو تتوقف قريباً جداً.
    'ada': 'cardano',
    'sol': 'solana',
    'doge': 'dogecoin',
    'matic': 'polygon',
    'dot': 'polkadot',
    'dai': 'dai',
    'shib': 'shiba-inu',
    'trx': 'tron',
    'avax': 'avalanche-2',
    'uni': 'uniswap',
    'wbtc': 'wrapped-bitcoin',
    'leo': 'leo-token',
    'ltc': 'litecoin'
}

# --- دالة جديدة: لجلب البيانات من CoinGecko API ---
def fetch_crypto_data_from_coingecko(coin_symbol, start_date_str, end_date_str):
    """
    تجلب بيانات الأسعار والحجم لعملة مشفرة واحدة من CoinGecko API لنطاق زمني محدد.

    Args:
        coin_symbol (str): رمز العملة المشفرة (مثل 'btc', 'eth').
        start_date_str (str): تاريخ البدء بتنسيق 'DD-MM-YYYY'.
        end_date_str (str): تاريخ الانتهاء بتنسيق 'DD-MM-YYYY'.

    Returns:
        pd.DataFrame: إطار بيانات (DataFrame) يحتوي على أعمدة 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'.
                      ملاحظة هامة: 'Open', 'High', 'Low' قد تكون قيمًا تقديرية أو NaN
                      لأن نقطة نهاية market_chart/range المجانية لـ CoinGecko
                      توفر بشكل أساسي سعر الإغلاق والحجم للبيانات اليومية المجمعة.
                      أو None إذا فشل جلب البيانات.
    """
    # الحصول على معرف CoinGecko للعملة
    coin_id = COINGECKO_IDS.get(coin_symbol.lower())
    if not coin_id:
        print(f"خطأ: لا يوجد معرّف CoinGecko لـ {coin_symbol}. يرجى إضافته إلى القائمة COINGECKO_IDS في السكربت.")
        return None

    print(f"جاري محاولة جلب البيانات لـ {coin_symbol.upper()} من {start_date_str} إلى {end_date_str} باستخدام CoinGecko API...")

    try:
        # تحويل التواريخ النصية (DD-MM-YYYY) إلى Unix Timestamps (ثوانٍ منذ Epoch)
        # CoinGecko يتوقع timestamps بالثواني، ويفضل UTC.
        from datetime import datetime, timedelta, timezone # تأكد من استيراد هذه المكتبات
        import requests # تأكد من استيراد requests

        start_dt_obj = datetime.strptime(start_date_str, '%d-%m-%Y').replace(tzinfo=timezone.utc)
        end_dt_obj = datetime.strptime(end_date_str, '%d-%m-%Y').replace(tzinfo=timezone.utc)

        # لضمان تضمين البيانات حتى نهاية اليوم المحدد في `end_date_str`،
        # نضيف يومًا واحدًا إلى تاريخ الانتهاء عند إرسال الطلب إلى API.
        # سيتم بعد ذلك تقليم البيانات بدقة باستخدام التواريخ الفعلية.
        end_dt_for_api = end_dt_obj + timedelta(days=1)

        start_timestamp = int(start_dt_obj.timestamp())
        end_timestamp = int(end_dt_for_api.timestamp())

        # بناء عنوان URL ومعاملات الطلب (Parameters) لـ CoinGecko
        url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart/range"
        params = {
            'vs_currency': 'usd', # العملة المقابلة (دائماً دولار أمريكي هنا)
            'from': start_timestamp,
            'to': end_timestamp
        }

        # --- أسطر الطباعة الجديدة للتشخيص ---
        print(f"URL being requested: {url}")
        print(f"Params being sent: {params}")
        # --- نهاية أسطر الطباعة الجديدة ---

        # لا حاجة لـ API key أو headers خاصة بالـ API في الطبقة المجانية لـ CoinGecko

        # أضف تأخيرًا صغيرًا لتجنب تجاوز حدود الطلبات (Rate Limiting)
        import time # تأكد من استيراد time
        time.sleep(13)

        response = requests.get(url, params=params)
        response.raise_for_status() # إطلاق استثناء لأخطاء HTTP (4xx أو 5xx)

        #import json # تأكد من استيراد json
        import pandas as pd # تأكد من استيراد pandas

        json_data = response.json()

        # CoinGecko يعيد 'prices' و 'total_volumes'
        # كل منهما عبارة عن قائمة من [timestamp_ms, value]
        prices_data = json_data.get('prices', [])
        volumes_data = json_data.get('total_volumes', [])

        if not prices_data:
            print(f"لم يتم العثور على بيانات لـ {coin_symbol.upper()} في استجابة CoinGecko API للنطاق الزمني المحدد.")
            return None

        # تحويل البيانات إلى DataFrame مؤقتة
        # الطوابع الزمنية من CoinGecko تكون بالمللي ثانية، لذا نستخدم unit='ms'
        df_prices = pd.DataFrame(prices_data, columns=['timestamp', 'close'])
        df_volumes = pd.DataFrame(volumes_data, columns=['timestamp', 'volume'])

        # تحويل الطوابع الزمنية إلى كائنات تاريخ (date)
        df_prices['date'] = pd.to_datetime(df_prices['timestamp'], unit='ms').dt.date
        df_volumes['date'] = pd.to_datetime(df_volumes['timestamp'], unit='ms').dt.date

        # دمج البيانات بناءً على التاريخ
        # نستخدم 'outer' لضمان الاحتفاظ بجميع التواريخ إذا كان هناك عدم تطابق بسيط (غير محتمل هنا)
        df_merged = pd.merge(
            df_prices[['date', 'close']],
            df_volumes[['date', 'volume']],
            on='date',
            how='outer'
        )
        # توليد الأعمدة Open, High, Low بنفس قيمة الإغلاق (close) كقيمة تقريبية
        df_merged['open'] = df_merged['close']
        df_merged['high'] = df_merged['close']
        df_merged['low'] = df_merged['close']

        # إعادة ترتيب الأعمدة وإعادة تعيين الفهرس
        df_final = df_merged.reset_index()
        df_final['Date'] = df_final['date'].apply(lambda d: d.strftime('%d-%m-%Y'))
        df_final['Coin'] = coin_symbol.upper()

        df_final = df_final[['Date', 'open', 'high', 'low', 'close', 'volume', 'Coin']]
        df_final.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Coin']

        return df_final

    except requests.exceptions.RequestException as e:
        print(f"خطأ في جلب البيانات لـ {coin_symbol.upper()} من CoinGecko API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"محتوى الاستجابة: {e.response.text}")
        return None
    except Exception as e:
        print(f"حدث خطأ غير متوقع لـ {coin_symbol.upper()}: {e}")
        return None

        df_merged.set_index('date', inplace=True)
        df_merged.sort_index(inplace=True)

        # قص إطار البيانات ليطابق تاريخ الانتهاء المحدد بدقة
        # (بما أننا ربما جلبنا بيانات ليوم إضافي لضمان تغطية تاريخ الانتهاء بالكامل من API)
        # ملاحظة: .loc[start_date_obj.date():end_date_obj.date()] يعمل بشكل أفضل بعد أن يكون الفهرس من نوع date
        df_final = df_merged.loc[start_dt_obj.date():end_dt_obj.date()]

        # ملء أعمدة 'Open', 'High', 'Low'
        # بما أن market_chart/range لا يوفرها مباشرة لليومية،
        # سنقوم بملئها بقيم تقديرية أو NaN. يفضل استخدام NaN إذا كنت تحتاج للدقة.
        # إذا قمت بتعيينها لـ NaN، فستحتاج إلى التعامل مع القيم المفقودة لاحقًا في التحليل.
        df_final.loc[:, 'open'] = df_final['close'] # أو pd.NA أو np.nan
        df_final.loc['high'] = df_final['close'] # أو pd.NA أو np.nan
        df_final.loc['low'] = df_final['close']  # أو pd.NA أو np.nan

        # إعادة تسمية الأعمدة وتغيير ترتيبها لتطابق التنسيق المتوقع (OHLCV)
        df_final.reset_index(inplace=True)
        df_final.rename(columns={'date': 'Date',
                                  'open': 'Open',
                                  'high': 'High',
                                  'low': 'Low',
                                  'close': 'Close',
                                  'volume': 'Volume'}, inplace=True)

        df_final = df_final.copy[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
#df_final = df_final.copy()

        return df_final

    except requests.exceptions.HTTPError as http_err:
        print(f"خطأ HTTP في جلب البيانات لـ {coin_symbol.upper()} من CoinGecko API: {http_err}")
        print(f"محتوى الاستجابة: {response.text}")
        return None
    except json.JSONDecodeError as json_err:
        print(f"خطأ في تحليل استجابة JSON من CoinGecko لـ {coin_symbol.upper()}: {json_err}")
        if 'response' in locals() and response is not None:
            print(f"المحتوى المستلم (أول 500 حرف): {response.text[:500]}")
        return None
    except Exception as err:
        print(f"حدث خطأ غير متوقع في جلب البيانات لـ {coin_symbol.upper()} من CoinGecko API: {err}")
        return None

# --- نهاية دالة CoinGecko الجديدة ---

#---
## تحديث محلل الوسائط المنطق الرئيسي

#```python
# الوسائط الافتراضية
now = datetime.now()
before = now - timedelta(days=1)

# محلل سطر الأوامر (CLI)
parser = argparse.ArgumentParser(description='إنشاء مجموعة بيانات أسعار OHLCV للعملات المشفرة.')

# المسار
parser.add_argument('-p', '--path', type=str, nargs='?', default=os.getcwd(),
                    help='مسار حفظ مجموعة البيانات (الافتراضي هو الدليل الحالي)')

# اسم الملف
parser.add_argument('-f', '--filename', type=str, nargs='?',
                    help='اسم ملف مجموعة البيانات (الافتراضي هو dataset_SOURCE_START_END.csv)')

# العملات
parser.add_argument('-c', '--coins', type=str, nargs='?',
                    help='مسار ملف JSON يحتوي على قائمة من رموز العملات (مثلاً src/examples/coins.json).')

# تاريخ البدء
parser.add_argument('-s', '--start', type=str, nargs='?', default=before.strftime('%d-%m-%Y'),
                    help='تاريخ بدء مجموعة البيانات (الافتراضي هو الأمس)')

# تاريخ الانتهاء
parser.add_argument('-e', '--end', type=str, nargs='?', default=now.strftime('%d-%m-%Y'),
                    help='تاريخ انتهاء مجموعة البيانات (الافتراضي هو اليوم)')

# مصدر البيانات الجديد (CoinMarketCap أو CoinGecko)
parser.add_argument('--source', type=str, choices=['CoinMarketCap', 'CoinGecko'], default='CoinGecko',
                    help='مصدر البيانات التاريخية (اختر: CoinMarketCap أو CoinGecko). الافتراضي هو CoinGecko.')

# تحليل الوسائط
args = parser.parse_args()
dfs = []
start = args.start
end = args.end
source = args.source # جلب مصدر البيانات المختار

# تحديد دالة جلب البيانات بناءً على المصدر
if source == 'CoinMarketCap':
    fetch_function = fetch_crypto_ohlc_data_from_api
    default_filename_prefix = 'dataset_coinmarketcap'
elif source == 'CoinGecko':
    fetch_function = fetch_crypto_data_from_coingecko
    default_filename_prefix = 'dataset_coingecko'
else:
    # هذا الشرط لن يتم الوصول إليه أبدًا بسبب 'choices' في argparse، ولكنه جيد كحماية
    print("مصدر بيانات غير صالح. يرجى الاختيار بين 'CoinMarketCap' أو 'CoinGecko'.")
    sys.exit(1)


# التحقق من اسم الملف
if not args.filename:
    # استخدام تنسيق ISO لتجنب التضارب إذا كانت التواريخ بتنسيق DD-MM-YYYY
    start_for_filename = datetime.strptime(start, '%d-%m-%Y').strftime('%Y%m%d')
    end_for_filename = datetime.strptime(end, '%d-%m-%Y').strftime('%Y%m%d')
    filename = f'{default_filename_prefix}_{start_for_filename}_{end_for_filename}.csv'
else:
    # التأكد من أن اسم الملف ينتهي بـ .csv
    if not args.filename.lower().endswith('.csv'):
        filename = args.filename + '.csv'
    else:
        filename = args.filename

# التحقق من التواريخ
try:
    # تحويل التواريخ النصية إلى كائنات datetime للتحقق
    start_dt = datetime.strptime(start, '%d-%m-%Y')
    end_dt = datetime.strptime(end, '%d-%m-%Y')

    # استثناء (start>end)
    if start_dt > end_dt:
        print('تم توفير تواريخ غير صالحة: وقت الانتهاء يسبق وقت البدء!')
        sys.exit(1)

    # التحويل مرة أخرى إلى نص لدالة API إذا لزم الأمر بهذا التنسيق،
    # أو تمرير كائنات datetime مباشرة إذا كانت الدالة الجديدة تفضلها.
    # (الآن يتم تمرير start و end كما هي لـ functions)


    # استثناء (مسار غير صالح)
    if not os.path.exists(args.path):
        print('تم توفير مسار غير صالح: الوجهة غير موجودة!')
        sys.exit(1)

    # إذا تم توفير ملف JSON للعملات المشفرة
    if args.coins:
        try:
            with open(args.coins, 'r') as f:
                coins_data = json.load(f)

            if isinstance(coins_data, list) and all(isinstance(c, str) for c in coins_data):
                cryptos = coins_data
            else:
                print('تنسيق بيانات غير صالح: تم توفير قائمة عملات غير صحيحة في ملف JSON!')
                sys.exit(1)

        except FileNotFoundError:
            print(f'تم توفير مسار غير صالح: ملف JSON للعملات ({args.coins}) غير موجود!')
            sys.exit(1)
        except json.JSONDecodeError:
            print(f'تنسيق بيانات غير صالح: ملف قائمة العملات ({args.coins}) ليس JSON صالحًا!')
            sys.exit(1)
    else: # وإلا إنشاء قائمة العملات المشفرة الافتراضية
        cryptos = ['btc', 'eth', 'usdt', 'usdc', 'bnb', 'xrp', 'busd', 'ada', 'sol', 'doge', 'matic', 'dot', 'dai', 'shib',
                   'trx', 'avax', 'uni', 'wbtc', 'leo', 'ltc']

    # طباعة الوسائط
    print({
        '--path': args.path,
        '--filename': filename,
        '--coins': cryptos,
        '--start': start,
        '--end': end,
        '--source': source
    })

    # حلقة الجلب - تستخدم الآن الدالة المحددة (fetch_function)
    for crypto in cryptos:
        df = fetch_function(crypto, start, end) # استخدم الدالة المحددة هنا
        if df is not None and not df.empty:
            if 'Coin' not in df.columns: # أضف عمود 'Coin' إذا لم يكن موجودًا بالفعل من الدالة
                df['Coin'] = crypto.upper()
            dfs.append(df)
        else:
            print(f"تعذر جلب البيانات لـ {crypto}. جاري التخطي.")

    # إنشاء إطار البيانات (df) (فقط إذا تم جلب أي بيانات)
    if dfs:
        df_final = pd.concat(dfs, ignore_index=True)

        # تنظيف العملات التي تحتوي على بيانات مفقودة
        if 'Date' in df_final.columns:
            df_final['Date'] = pd.to_datetime(df_final['Date'], format='%d-%m-%Y', errors='coerce')
            df_final.dropna(subset=['Date'], inplace=True)

            last_dt = end_dt
            first_dt = start_dt
            n_days = (last_dt - first_dt).days + 1
            if n_days <= 0: n_days = 1

            v_count = df_final.groupby('Coin').size()
            to_remove = v_count[v_count < n_days].index
            df_final = df_final[~df_final.Coin.isin(to_remove)]
            df_final.reset_index(drop=True, inplace=True)

            df_final['Date'] = df_final['Date'].dt.strftime('%Y-%m-%d') # تغيير التنسيق إلى YYYY-MM-DD
            # التأكد من ترتيب الأعمدة
            required_cols = ['Coin', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df_final = df_final[required_cols]


# ... (بقية كود src/data_pull.py، بما في ذلك دالة main() وتجميع df_final) ...

    if __name__ == '__main__':
        args = docopt(__doc__)
    # ... (بقية المنطق الرئيسي الذي يجمع البيانات في df_final) ...


    if not df_final.empty:
        # 1. تحديد مجلد الإخراج الثابت الذي نرغب في حفظ الملف فيه.
        fixed_output_dir = "data"

        # 2. التأكد من إنشاء مجلد الإخراج الثابت إذا لم يكن موجودًا بالفعل.
        if not os.path.exists(fixed_output_dir):
            os.makedirs(fixed_output_dir)

        # 3. تحديد اسم الملف الثابت (ohlc.csv)
        fixed_filename = "ohlc.csv"

        # 4. بناء المسار الكامل لحفظ الملف باستخدام المجلد والاسم الثابتين.
        full_fixed_path = os.path.join(fixed_output_dir, fixed_filename)

        # 5. حفظ الـ DataFrame النهائي إلى المسار الجديد الثابت.
        #    هذا السطر سيحل محل السطر الأصلي للحفظ (مثلاً: df_final.to_csv(file_name_with_path, ...))
        df_final.to_csv(full_fixed_path, sep=',', encoding='utf-8', index=False)

        # 6. طباعة رسالة تأكيد للمستخدم بالمسار الجديد.
        print(f"تم حفظ مجموعة البيانات في: {full_fixed_path}")
    else:
        print("لم يتم جلب بيانات لأي عملة. لم يتم إنشاء ملف CSV.")


except ValueError as e:
    print(f'حدث خطأ أثناء معالجة التاريخ أو تنظيف البيانات: {e}')
    print('الرجاء التحقق من تنسيقات التواريخ وهيكل البيانات التي أرجعتها واجهة برمجة التطبيقات (API).')
    sys.exit(1)
except Exception as e:
    print(f"حدث خطأ غير متوقع: {e}")
    sys.exit(1)