# feature_engineering.py (النسخة النهائية - بمنطق دمج مبسط)

import pandas as pd
import pandas_ta as ta
import sys
import argparse

def create_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    الدالة الكاملة التي تقوم بكل عمليات هندسة الميزات بمنطق دمج مبسط.
    """
    print("--- بدء عملية هندسة الميزات الكاملة ---")
    
    df = raw_df.copy()

    # التأكد من وجود الأعمدة الأساسية
    if 'Coin' not in df.columns or 'Date' not in df.columns:
        raise ValueError("البيانات الخام يجب أن تحتوي على عمودي 'Coin' و 'Date'.")
        
    # تحويل عمود التاريخ إلى كائنات تاريخ إذا لم يكن كذلك بالفعل
    df['Date'] = pd.to_datetime(df['Date'])
    
    # إنشاء DataFrame نهائي فارغ مع فهرس التاريخ الصحيح
    unique_dates = df['Date'].unique()
    final_df = pd.DataFrame(index=pd.to_datetime(unique_dates))
    final_df.sort_index(inplace=True)

    coins = df['Coin'].unique()
    
    for coin in coins:
        print(f"  - حساب الميزات لعملة {coin.upper()}...")
        
        # فلترة البيانات للعملة الحالية وتعيين التاريخ كفهرس
        coin_df = df[df['Coin'] == coin].set_index('Date').copy()
        
        # استخدام pandas_ta لحساب المؤشرات
        # ملاحظة: تأكد من أن هذه هي نفس المؤشرات والإعدادات التي تم تدريب النموذج عليها
        coin_df.ta.atr(length=14, append=True)
        coin_df.ta.bbands(length=5, std=2.0, append=True) # استخدام length=5 كما هو واضح من الخطأ السابق
        coin_df.ta.macd(append=True)
        coin_df.ta.rsi(length=14, append=True)
        coin_df.ta.sma(length=7, append=True)
        
        # حساب متوسط السعر OHLC
        coin_df[f"{coin.lower()}_avg_ohlc"] = coin_df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
        coin_df[coin.upper()] = coin_df['Close'] # عمود السعر نفسه
        coin_df[f"{coin.lower()}_volume"] = coin_df['Volume'] # عمود حجم التداول

        # إعادة تسمية الأعمدة التي أنشأتها pandas_ta
        rename_dict = {
            'ATRr_14': f"{coin.lower()}_atr",
            'BBL_5_2.0': f"{coin.lower()}_bbl",
            'BBM_5_2.0': f"{coin.lower()}_bbm",
            'BBU_5_2.0': f"{coin.lower()}_bbu",
            'BBB_5_2.0': f"{coin.lower()}_bbb",
            'BBP_5_2.0': f"{coin.lower()}_bbp",
            'MACD_12_26_9': f"{coin.lower()}_macd",
            'MACDh_12_26_9': f"{coin.lower()}_macdh",
            'MACDs_12_26_9': f"{coin.lower()}_macds",
            'RSI_14': f"{coin.lower()}_rsi14",
            'SMA_7': f"{coin.lower()}_sma7"
        }
        coin_df.rename(columns=rename_dict, inplace=True)
        
        # اختيار الأعمدة التي نريد إضافتها فقط إلى الجدول النهائي
        # (كل الأعمدة التي تبدأ برمز العملة)
        features_to_join = [col for col in coin_df.columns if col.startswith(coin.lower()) or col == coin.upper()]
        
        # استخدام join لدمج الميزات الجديدة مع الجدول النهائي
        final_df = final_df.join(coin_df[features_to_join])

    # إزالة أي صفوف تحتوي على قيم فارغة بعد كل عمليات الدمج والحساب
    final_df.dropna(inplace=True)
    
    print("\nاكتملت هندسة الميزات الكاملة.")
    return final_df


def main():
    """
    الدالة الرئيسية لتمكين تشغيل السكربت من سطر الأوامر.
    """
    parser = argparse.ArgumentParser(description='إنشاء ميزات فنية من بيانات OHLC.')
    parser.add_argument('-d', '--data', type=str, required=True, help='مسار ملف بيانات OHLC الأولية.')
    parser.add_argument('-o', '--output', type=str, default='data/features.csv', help='مسار حفظ ملف الميزات الجديد.')
    args = parser.parse_args()

    print(f"جاري تحميل البيانات الأولية من: {args.data}")
    try:
        raw_df_from_file = pd.read_csv(args.data, parse_dates=['Date'])
    except FileNotFoundError:
        print(f"خطأ: لم يتم العثور على ملف البيانات في المسار المحدد: {args.data}"); sys.exit(1)

    final_features_df = create_features(raw_df_from_file)
    
    if not final_features_df.empty:
        final_features_df.to_csv(args.output)
        print(f"تم حفظ البيانات مع الميزات الجديدة في: {args.output}")
    else:
        print("لم يتم إنشاء ملف الميزات لأن DataFrame الناتج فارغ.")


if __name__ == '__main__':
    main()