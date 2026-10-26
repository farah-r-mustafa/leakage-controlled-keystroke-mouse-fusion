# Keystroke–Mouse Stage 3A

## الغرض

يبني هذا الـPilot ملفات Genuine وImpostor Claims من مخرجات Stage 2C المجمدة، دون تدريب نماذج أو حساب درجات أو اختيار عتبات.

## الإدخال

ملف:

`keystroke_mouse_stage2c_review.zip`

لا تستخدم ملف Dataset الأصلي في هذه المرحلة.

## أمر التشغيل المباشر

```bat
python "<local_path>" --input-stage2c "<local_path>" --output "<local_path>"
```

أو:

```bat
py -3.12 "<local_path>" --input-stage2c "<local_path>" --output "<local_path>"
```

عند وجود مجلد أو ZIP سابق:

```bat
python "<local_path>" --input-stage2c "<local_path>" --output "<local_path>" --overwrite
```

## المتطلبات

- Windows 10
- Python 3.12
- NumPy

لتثبيت NumPy عند الحاجة:

```bat
python -m pip install numpy
```

## البروتوكولات

### Balanced Genuine–Impostor 1:1

لكل مستخدم مطالب به، يحتفظ السكربت بجميع النوافذ الحقيقية ويختار عدداً مساوياً من نوافذ المستخدمين الآخرين دون استبدال. يتم التوزيع قدر الإمكان بين المستخدمين والجلسات، باستخدام ترتيب حتمي وبذرة مجمدة.

### Realistic All-Impostor Claims

تُختبر كل نافذة ضد المستخدمين الأحد عشر. لذلك تنتج كل نافذة مطالبة حقيقية واحدة وعشر مطالبات انتحال.

## المخرجات

ينشئ السكربت مجلداً وملف ZIP يحتويان على:

- 24 ملف Claims: أربعة مسارات × بروتوكولين × ثلاثة أقسام.
- ملفات Validation وTest المستقلة.
- `source_row_index` للربط المباشر مع مصفوفات Stage 2C.
- `claimed_user_id`, `actual_user_id`, `is_genuine`.
- تدقيق أخذ عينات الانتحال.
- تدقيق سلامة الـClaims والتداخل بين الأقسام.
- بصمات SHA-256.
- تقرير التحذيرات الخاصة بالخلايا قليلة النوافذ الحقيقية.

## قيود المرحلة

لا ينفذ السكربت:

- تدريب النماذج.
- حساب الدرجات.
- Temperature Scaling.
- اختيار العتبات.
- حساب المقاييس.
- أي استخدام لنتائج Test في القرارات.

بعد التنفيذ ارفع الملف:

`<local_path>
