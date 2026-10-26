# Keystroke–Mouse Stage 3B Classical Pilot v3.1.2

هذه نسخة مخصصة لإعادة التدريب الكامل على Windows بعد إصلاح خطأ `WinError 32` الذي كان يظهر عند تنظيف ملفات NPZ المؤقتة.

## الإصلاحات

- إغلاق ملفات NPZ صراحةً بعد نسخ المصفوفات المطلوبة.
- تحرير مصفوفات التدريب والاختبار والنماذج من الذاكرة بعد كل Cohort.
- إعادة محاولة حذف المجلد المؤقت خمس مرات بفواصل متدرجة.
- عدم السماح لقفل Windows المؤقت بإلغاء النتائج المكتملة.
- ضغط ZIP سريع وآمن لتقليل مدة الخطوة النهائية.
- فحص SHA-256 لكل ملف ناتج قبل إنشاء ZIP.
- فحص CRC وعدد الملفات داخل ZIP بعد إنشائه.
- إنشاء `STAGE3B_COMPLETE.ok` فقط بعد اكتمال التدريب والدرجات.

## المدخلات

- `keystroke_mouse_stage2c_review.zip`
- `keystroke_mouse_stage3a_review.zip`

لا تستخدم ملف Dataset الأصلي في هذه المرحلة.

## تثبيت المكتبات عند الحاجة

```bat
py -3.12 -m pip install -r requirements.txt
```

## التشغيل المباشر

```bat
py -3.12 keystroke_mouse_stage3b_classical_pilot.py --input-stage2c "<local_path>" --input-stage3a "<local_path>" --output "<local_path>" --overwrite
```

يمكن أيضاً تشغيل:

```text
run_stage3b_full_retrain.bat
```

> ملاحظة: الخيار `--overwrite` يحذف مجلد وZIP Stage 3B السابقين قبل إعادة التدريب. أغلق أي CSV أو ZIP سابق مفتوح في Excel أو File Explorer قبل التشغيل.

## علامات الاكتمال الصحيح

يجب أن ينتج:

```text
<local_path>
<local_path>
```

ويجب أن يحتوي المجلد على:

```text
STAGE3B_COMPLETE.ok
output_manifest_sha256.csv
stage3b_summary.json
execution_log.txt
```

وفي نهاية CMD يجب أن يظهر:

```text
"manifest_files_verified": 251
"zip_crc_verified": true
```

الحالة المتوقعة هي `PASS_WITH_WARNINGS` بسبب ندرة نوافذ بعض المستخدمين، وليست بسبب فشل التنفيذ.
