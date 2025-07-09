# check_features.py
import json

def verify_features():
    """
    يقوم هذا السكربت بمقارنة الميزات المطلوبة مع الميزات المرسلة
    ويطبع أي ميزات مفقودة.
    """
    try:
        # تحميل قائمة الميزات المطلوبة التي يتوقعها النموذج
        with open('config/features.json', 'r') as f:
            required_data = json.load(f)
        required_features = set(required_data['features'])

        # تحميل البيانات التجريبية التي سترسلها للـ API
        with open('test_payload.json', 'r') as f:
            payload_data = json.load(f)
        
        # التأكد من أن payload يحتوي على مفتاح 'sequence' وأن به بيانات
        if 'sequence' not in payload_data or not payload_data['sequence']:
            print("Error: 'test_payload.json' is empty or badly formatted.")
            return

        # الحصول على أسماء الميزات من أول سجل في التسلسل المرسل
        provided_features = set(payload_data['sequence'][0].keys())

        # --- المقارنة ---
        missing_features = required_features - provided_features
        extra_features = provided_features - required_features

        print("-" * 50)
        if not missing_features:
            print("✅ تهانينا! كل الميزات المطلوبة موجودة في ملف test_payload.json.")
        else:
            print("❌ خطأ: هناك ميزات مطلوبة غير موجودة في test_payload.json:")
            for feature in sorted(list(missing_features)):
                print(f"  - {feature}")
        
        if extra_features:
            print("\n⚠️ ملاحظة: هناك ميزات إضافية في test_payload.json غير مطلوبة من النموذج:")
            for feature in sorted(list(extra_features)):
                print(f"  - {feature}")
        print("-" * 50)

    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. Make sure 'features.json' and 'test_payload.json' exist.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# تشغيل الدالة
if __name__ == "__main__":
    verify_features()