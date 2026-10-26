# Keystroke–Mouse Stage 2B Pilot

## الغرض

تنفيذ مرحلة **Window Materialization and Leakage-Safe Feature Engineering** فقط، اعتماداً على التقسيم المجمد المقبول من Stage 2A.

لا ينفذ السكربت أي تدريب أو Claims أو تطبيع أو اختيار ميزات أو معايرة أو عتبات.

## متطلبات التشغيل

- Windows 10
- Python 3.12
- لا توجد مكتبات خارجية مطلوبة
- يفضّل توفر مساحة حرة لا تقل عن 500 MB لمجلد النتائج المؤقت والنهائي

## أمر التشغيل المباشر

افتح CMD داخل مجلد الحزمة، ثم نفّذ:

```bat
python keystroke_mouse_stage2b_pilot.py --input "<local_path>" --output "<local_path>"
```

أو:

```bat
py -3.12 keystroke_mouse_stage2b_pilot.py --input "<local_path>" --output "<local_path>"
```

عند وجود مجلد إخراج سابق أضف:

```text
--overwrite
```

## ملفات Stage 2A

لا حاجة لتمرير مسار Stage 2A. تحتوي الحزمة داخل `stage2a_frozen_inputs` على النسخ المقبولة من:

- `frozen_session_manifest.csv`
- `primary_segment_manifest_60s.csv`
- `semantic_duplicate_groups.csv`

يتوقف السكربت إذا لم تتطابق بصماتها مع البصمات المجمدة.

## المخرجات الأساسية

- جداول Features منفصلة لـTrain وValidation وTest.
- تمثيلات تسلسلية JSONL لـCNN وBiLSTM والنموذج الهجين.
- `window_manifest.csv` مع Window IDs ثابتة.
- فهرس النوافذ متعددة الوسائط الأساسية `eligible_both`.
- Feature Dictionary وFeature Provenance.
- تدقيق Missing/Infinity/Outliers باستخدام حدود مشتقة من Training فقط وللتدقيق دون قص القيم.
- تقارير التغطية والتسرب والبصمات وسجل التنفيذ.
- ملف ZIP نهائي بجانب مجلد الإخراج.

## قرار التوقف

بعد التنفيذ ارفع فقط:

```text
<local_path>
```

ولا تبدأ التدريب قبل مراجعة الحزمة.
