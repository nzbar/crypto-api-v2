# ----------------------------------------------------
# Dockerfile لتطبيق Flask مع خادم Gunicorn
# (النسخة النهائية المصححة والمحسّنة)
# ----------------------------------------------------

# الخطوة 1: ابدأ من صورة بايثون رسمية وخفيفة
FROM python:3.11.9-slim

# الخطوة 2: تجهيز بيئة العمل
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app

# الخطوة 3: انسخ ملف المتطلبات أولاً للاستفادة من التخزين المؤقت
COPY requirements.txt .

# الخطوة 4: تثبيت أدوات الترجمة والمكتبات ثم تنظيف المخلفات
# يتم كل هذا في خطوة RUN واحدة لتقليل حجم الصورة النهائية
RUN apt-get update && \
    apt-get install -y build-essential && \
    \
    # الآن بعد تثبيت أدوات البناء، نقوم بتثبيت مكتبات بايثون
    pip install --no-cache-dir -r requirements.txt && \
    \
    # بعد الانتهاء، نحذف أدوات الترجمة التي لم نعد بحاجة إليها
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

# الخطوة 5: انسخ باقي كود المشروع
# سيتم نسخ كل شيء ما عدا الملفات المذكورة في .dockerignore
COPY . .

# الخطوة 6: تشغيل الخادم مع الإعدادات المحسّنة
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "3", "--timeout", "120", "app:app"]