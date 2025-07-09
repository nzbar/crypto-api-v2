"""
File: evaluate_forecast.py
Description: تحليل ورسم النتائج البيانية للتنبؤات مقارنة بالبيانات التاريخية.
File Created: 07/06/2025
Python Version: 3.9
"""
import os
import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

def parse_arguments():
    """تحليل وسيطات سطر الأوامر."""
    parser = argparse.ArgumentParser(description='تحليل ورسم نتائج التنبؤ.')
    
    parser.add_argument('-p', '--predictions', type=str, required=True,
                        help='مسار ملف التنبؤات النصي (e.g., predictions_lstm_BTC_20250607.txt).')
    
    parser.add_argument('-d', '--historical_data', type=str, required=True,
                        help='مسار ملف CSV الذي يحتوي على البيانات التاريخية للتحقق (e.g., data/split/valid_07062025.csv).')
                        
    parser.add_argument('-t', '--target', type=str, required=True,
                        help='العمود المستهدف الذي تم التنبؤ به (e.g., BTC).')

    parser.add_argument('-w', '--window', type=int, default=30,
                        help='عدد الأيام التاريخية التي ستظهر في الرسم البياني للمقارنة (الافتراضي: 30).')

    return parser.parse_args()

def main():
    """الدالة الرئيسية لتشغيل عملية التحليل والرسم."""
    args = parse_arguments()

    # --- 1. تحميل البيانات ---
    print("جاري تحميل البيانات التاريخية والتنبؤات...")
    try:
        # تحميل البيانات التاريخية
        history_df = pd.read_csv(args.historical_data, index_col='Date', parse_dates=True)
        
        # تحميل ملف التنبؤات
        predictions_array = np.loadtxt(args.predictions)
        
    except FileNotFoundError as e:
        print(f"خطأ: لم يتم العثور على الملف: {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"حدث خطأ أثناء تحميل البيانات: {e}")
        sys.exit(1)

    # --- 2. إعداد البيانات للرسم ---
    
    # أخذ آخر جزء من البيانات التاريخية للعرض
    historical_to_plot = history_df[args.target].tail(args.window)
    
    # إنشاء نطاق زمني للتنبؤات
    last_historical_date = historical_to_plot.index[-1]
    prediction_dates = pd.date_range(start=last_historical_date + timedelta(days=1), periods=len(predictions_array))
    
    # إنشاء DataFrame للتنبؤات
    predictions_df = pd.Series(predictions_array.flatten(), index=prediction_dates, name=f"تنبؤ {args.target}")

    # --- 3. إنشاء الرسم البياني ---
    print("جاري إنشاء الرسم البياني...")
    plt.style.use('seaborn-v0_8-darkgrid') # استخدام نمط جميل للرسم
    fig, ax = plt.subplots(figsize=(15, 8)) # تحديد حجم الرسم

    # رسم البيانات التاريخية
    ax.plot(historical_to_plot.index, historical_to_plot.values, label=f'السعر الفعلي لـ {args.target}', color='dodgerblue', marker='o', linestyle='-')
    
    # رسم البيانات المتوقعة
    ax.plot(predictions_df.index, predictions_df.values, label=f'السعر المتوقع لـ {args.target}', color='orangered', marker='o', linestyle='--')

    # --- 4. تخصيص شكل الرسم البياني ---
    
    # تحديد عنوان الرسم والمحاور
    fig.suptitle(f'تحليل أداء نموذج التنبؤ لعملة {args.target}', fontsize=20)
    ax.set_title(f'آخر {args.window} يومًا من البيانات الفعلية مقارنة بـ {len(predictions_df)} يومًا من التنبؤات', fontsize=12)
    ax.set_xlabel('التاريخ', fontsize=12)
    ax.set_ylabel('السعر (USD)', fontsize=12)
    
    # تدوير تواريخ المحور السيني لتكون واضحة
    plt.xticks(rotation=45)
    
    # إضافة شبكة خلفية
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # إضافة وسيلة الإيضاح (Legend)
    ax.legend(fontsize=12)

    # --- 5. عرض الرسم البياني ---
    print("اكتمل. سيتم الآن عرض الرسم البياني...")
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # ضبط الهوامش
    plt.show()

if __name__ == '__main__':
    main()

