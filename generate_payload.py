# generate_payload.py
import json
import random

# المسارات للملفات المصدر والهدف
FEATURES_CONFIG_PATH = 'config/features.json'
OUTPUT_PAYLOAD_PATH = 'test_payload.json'
SEQUENCE_LENGTH = 60 # عدد النقاط الزمنية التي يتوقعها النموذج

print(f"--- بدء إنشاء ملف حمولة اختبار صالح في '{OUTPUT_PAYLOAD_PATH}' ---")

# 1. تحميل قائمة الميزات المطلوبة
try:
    with open(FEATURES_CONFIG_PATH, 'r') as f:
        required_features = json.load(f)['features']
    print(f"تم العثور على {len(required_features)} ميزة مطلوبة.")
except Exception as e:
    print(f"خطأ فادح: لم يتم العثور على ملف الميزات أو لا يمكن قراءته: {e}")
    exit()

# 2. إنشاء بيانات عشوائية للتسلسل
sequence_data = []
for _ in range(SEQUENCE_LENGTH):
    # إنشاء قاموس (dictionary) لكل نقطة زمنية
    timestep_data = {}
    for feature in required_features:
        # توليد قيمة رقمية عشوائية لكل ميزة
        timestep_data[feature] = round(random.uniform(0.1, 100.0), 4)
    sequence_data.append(timestep_data)

print(f"تم توليد تسلسل بيانات يحتوي على {len(sequence_data)} نقطة زمنية.")

# 3. تكوين الحمولة النهائية وحفظها في ملف JSON
final_payload = {
    "sequence": sequence_data
}

try:
    with open(OUTPUT_PAYLOAD_PATH, 'w') as f:
        json.dump(final_payload, f, indent=2)
    print(f"✅ نجاح! تم إنشاء ملف '{OUTPUT_PAYLOAD_PATH}' بنجاح مع كل الميزات المطلوبة.")
except Exception as e:
    print(f"خطأ فادح: لم أتمكن من كتابة الملف. تفاصيل الخطأ: {e}")

print("\n--- يمكنك الآن استخدام هذا الملف لاختبار الـ API ---")