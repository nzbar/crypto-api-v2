"""
File: feature_engineering.py
Description: إنشاء ميزات فنية متقدمة من بيانات OHLC الأولية وتحويلها إلى الشكل العريض.
File Created: 07/06/2025
Python Version: 3.9
"""
import argparse
import pandas as pd
import sys

def calculate_rsi(series, period=14):
    """حساب مؤشر القوة النسبية (RSI)."""
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def main():
    """الدالة الرئيسية لإنشاء وحفظ الميزات."""
    parser = argparse.ArgumentParser(description='إنشاء ميزات فنية من بيانات OHLC.')
    parser.add_argument('-d', '--data', type=str, required=True, help='مسار ملف بيانات OHLC الأولية (e.g., data/ohlc.csv).')
    parser.add_argument('-o', '--output', type=str, default='data/features.csv', help='مسار حفظ ملف الميزات الجديد.')
    args = parser.parse_args()

    print(f"جاري تحميل البيانات الأولية من: {args.data}")
    try:
        df = pd.read_csv(args.data, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"خطأ: لم يتم العثور على ملف البيانات في المسار المحدد: {args.data}"); sys.exit(1)

    print("بدء عملية هندسة الميزات...")
    
    # --- الخطوة 1: حساب متوسط السعر وتحويل البيانات من الشكل الطولي إلى العريض ---
    print("  - حساب متوسط السعر (avg_ohlc) وتحويل البيانات إلى الشكل العريض...")
    df['avg_ohlc'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
    pivot_df = df.pivot(columns='Coin', values='avg_ohlc')
    
    # التأكد من أن أسماء الأعمدة كلها بأحرف صغيرة
    pivot_df.columns = [f"{col.lower()}_avg_ohlc" for col in pivot_df.columns]
    
    # --- الخطوة 2: حساب الميزات الفنية على البيانات العريضة ---
    features_list = [pivot_df] # قائمة لتجميع كل الميزات الجديدة
    coins = [col.replace('_avg_ohlc', '') for col in pivot_df.columns]
    
    for coin in coins:
        price_col = f'{coin}_avg_ohlc'
        print(f"  - حساب الميزات لعملة {coin.upper()}...")
        
        # إنشاء DataFrame مؤقت لميزات العملة الحالية
        coin_features = pd.DataFrame(index=pivot_df.index)
        
        # المتوسطات المتحركة
        coin_features[f'{coin}_sma_7'] = pivot_df[price_col].rolling(window=7).mean()
        coin_features[f'{coin}_sma_30'] = pivot_df[price_col].rolling(window=30).mean()
        coin_features[f'{coin}_ema_7'] = pivot_df[price_col].ewm(span=7, adjust=False).mean()
        coin_features[f'{coin}_ema_30'] = pivot_df[price_col].ewm(span=30, adjust=False).mean()
        
        # المؤشرات الفنية
        coin_features[f'{coin}_rsi_14'] = calculate_rsi(pivot_df[price_col], 14)
        coin_features[f'{coin}_roc_14'] = ((pivot_df[price_col] - pivot_df[price_col].shift(14)) / pivot_df[price_col].shift(14)) * 100
        coin_features[f'{coin}_volatility_14'] = pivot_df[price_col].rolling(window=14).std()

        features_list.append(coin_features)

    # دمج كل الميزات معًا مرة واحدة لتحسين الأداء
    all_features_df = pd.concat(features_list, axis=1)

    # إزالة الصفوف التي تحتوي على قيم فارغة (NaN) الناتجة عن الحسابات
    all_features_df.dropna(inplace=True)

    print("\nاكتملت هندسة الميزات.")
    
    # حفظ DataFrame الجديد
    all_features_df.to_csv(args.output)
    print(f"تم حفظ البيانات مع الميزات الجديدة في: {args.output}")
    
    print("\nعينة من أعمدة ملف الميزات الجديد للتأكيد:")
    print(all_features_df.head(2).to_markdown(tablefmt="grid"))


if __name__ == '__main__':
    main()
