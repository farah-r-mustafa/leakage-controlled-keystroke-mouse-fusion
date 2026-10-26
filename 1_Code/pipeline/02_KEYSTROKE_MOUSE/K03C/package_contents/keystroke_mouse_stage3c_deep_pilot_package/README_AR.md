# Keystroke–Mouse Stage 3C — Deep Models Pilot v3.2.0

## الغرض
تدريب CNN وBiLSTM وCNN+BiLSTM وإنتاج درجات خام على مستوى Claims المجمدة في Stage 3A.

## ما لا تنفذه الحزمة
- لا معايرة.
- لا اختيار عتبات.
- لا حساب نتائج Test النهائية.
- لا تغيير Claims أو تقسيم الجلسات.
- لا إعادة ملاءمة معالجة Stage 2C.

## المتطلبات
- Windows 10/11.
- Python 3.12.
- RAM 16 GB مناسبة.
- CPU فقط مدعوم.

تثبيت المتطلبات:

```bat
py -3.12 -m pip install numpy scikit-learn psutil
py -3.12 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## أمر التشغيل

```bat
py -3.12 "<local_path>" --input-stage2c "<local_path>" --input-stage3a "<local_path>" --output "<local_path>" --overwrite
```

أو عدّل المسارات في `run_stage3c_full.bat` ثم شغّله.

## المخرجات المتوقعة
- 132 نموذجاً: 4 مسارات × 3 معماريات × 11 مستخدماً.
- 48 ملف درجات: 4 مسارات × 3 معماريات × بروتوكولين × Validation/Test.
- `STAGE3C_COMPLETE.ok` بعد اكتمال جميع الخطوات فقط.
- `output_manifest_sha256.csv`.
- `keystroke_mouse_stage3c_review.zip`.

## حالة النجاح
يجب أن يتضمن الإخراج النهائي:

```text
manifest_files_verified: 192
zip_crc_verified: true
```

قد تكون الحالة `PASS_WITH_WARNINGS` بسبب قلة نوافذ بعض المستخدمين أو اقتصار Genuine Training على جلسة واحدة. هذه تحذيرات منهجية وليست أخطاء تنفيذ.
