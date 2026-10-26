# Keystroke–Mouse Stage 2C

## الهدف
تنفيذ المعالجة المسبقة باستخدام Training فقط، وتجميد مخططات الميزات والتنسورات قبل إنشاء Claims أو تدريب النماذج.

## المتطلبات
- Windows 10
- Python 3.12
- NumPy

لتثبيت NumPy عند الحاجة:

```bat
python -m pip install numpy
```

## ملف الإدخال
استخدم ملف نتائج Stage 2B المقبول:

```text
keystroke_mouse_stage2b_review.zip
```

لا تستخدم Dataset الأصلية مباشرة في هذه المرحلة.

## أمر التشغيل المباشر

```bat
python keystroke_mouse_stage2c_pilot.py --input-stage2b "<local_path>" --output "<local_path>"
```

إذا كان ملف Stage 2B داخل مجلد آخر، عدّل المسار فقط.

عند وجود مجلد إخراج سابق:

```bat
python keystroke_mouse_stage2c_pilot.py --input-stage2b "<local_path>" --output "<local_path>" --overwrite
```

## ما ينفذه السكربت
- التحقق من PASS وبصمات Stage 2B.
- Median imputation باستخدام Training فقط.
- Standard scaling باستخدام Training فقط.
- حذف الميزات الفارغة كلياً أو ثابتة التباين وفق Training فقط.
- إنشاء مصفوفات Tabular لـKeyboard وMouse وMultimodal-primary وMultimodal-any.
- إنشاء Tensor sequences وأقنعة Padding.
- تجميد ترتيب الميزات والقنوات وأطوال التسلسلات.
- حفظ user label mapping وlineage indexes.
- فحص القيم غير المنتهية والتداخل بين Train/Validation/Test.
- إنشاء ZIP تلقائياً للنتائج.

## ما لا ينفذه
- لا ينشئ Genuine/Impostor Claims.
- لا يدرب نماذج.
- لا يختار Thresholds.
- لا يستخدم Validation أوTest لملاءمة أي معالج.
- لا يجري Feature Selection اعتماداً على النتائج.

## الملف المطلوب رفعه بعد التشغيل

```text
<local_path>
```
