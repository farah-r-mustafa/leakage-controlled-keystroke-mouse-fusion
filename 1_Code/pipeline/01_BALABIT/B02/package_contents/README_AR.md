# Balabit Stage 2 — المعالجة المسبقة والنوافذ

## البداية الصحيحة

ابدأ حصراً بالملف:

```text
RUN_STAGE2_PILOT.bat
```

ضع مسار الداتاسيت الخام الذي يحتوي المجلدات مثل:

```text
training_files
test_files
fake_data
```

لا تضع مسار حزمة السكربت نفسها.

## ماذا يفعل الـPilot؟

- يختار حتى 5 جلسات تدريب لكل مستخدم.
- ينفذ تقسيم Session-Level إلى Train وValidation وInternal Test.
- ينشئ نافذتي 30 و50 حدثاً بتداخل 50%.
- يستخرج ميزات الحركة والزمن والسرعة والتسارع والـJerk والزاوية.
- يحفظ النوافذ بصيغة CSV.GZ مضغوطة.
- لا يدرب نموذجاً ولا يعدل الملفات الخام.

بعد النجاح ارفع:

```text
balabit_stage2_pilot_results.zip
```

بعد مراجعة الـPilot شغّل:

```text
RUN_STAGE2_FULL.bat
```

## المخرجات

- `canonical_manifest_all.csv`
- `protocol_manifest.csv`
- `session_quality.csv`
- `session_errors.csv`
- `window_counts.csv`
- `feature_dictionary.csv`
- `window_features/window_features_w30.csv.gz`
- `window_features/window_features_w50.csv.gz`
- وفي التنفيذ الكامل: w70 وw90
- `stage2_summary.json`
- `run_log.txt`

## قرارات منهجية

- `fake_data` مستبعدة من التدريب الأساسي ومحجوزة لاختبار Mimicry/Robustness لاحقاً.
- ملفات `test_files` مفهرسة فقط ولا تستخدم في التقييم من دون ملف تسميات رسمي.
- التقسيم يتم على الجلسات، قبل النوافذ.
- لا توجد نافذة تتجاوز حدود الجلسة.
- لا يطبق StandardScaler الآن؛ سيطبق على Train فقط في مرحلة النماذج.
- التكرارات تبقى افتراضياً؛ حذفها سيكون تجربة Ablation مستقلة عند الحاجة.
