# Keystroke–Mouse Stage 5.1 Pilot

حزمة إحصائية آمنة تقرأ مخرجات Stage 4B المجمّدة فقط. لا تعيد التدريب، ولا المعايرة، ولا اختيار العتبات، ولا تعدّل ملفات الإدخال.

## ما تنفذه

- McNemar على Claims المتطابقة بين Random Forest وكل من CNN وBiLSTM وCNN+BiLSTM.
- Bootstrap عنقودي مقترن على مستوى الجلسة، مع stratification حسب actual user.
- فواصل ثقة لـ AUC وEER وFAR وFRR وF1 وMCC.
- Exact paired sign-flip على مستوى claimed user لمقارنات النماذج والوسائط والبروتوكولات.
- Holm correction داخل كل عائلة اختبارات، مع Holm عالمي إضافي.
- Exact user-level label permutation لتحليل Adequate مقابل Limited وPersonal threshold مقابل Global fallback.
- Scorecard ترشيح نشر يجمع الأداء والدلالة والكلفة، ويظل ترشيح Pilot مؤقتاً حتى Full mode.
- ملفات تصدير موحّدة جاهزة للمقارنة اللاحقة مع Balabit Stage 5.1، دون تنفيذ تلك المقارنة الآن.

## المتطلبات

```text
Python 3.10+
pip install -r requirements_stage5_1.txt
```

## تشغيل Pilot

```bat
RUN_STAGE5_1_PILOT.bat "<local_path>" "<local_path>"
```

أو مباشرة:

```bat
python keystroke_mouse_stage5_1_pilot.py --input "<local_path>" --output "<local_path>" --mode pilot --bootstrap-reps 100 --seed 5102026
```

## تشغيل Full بعد نجاح Pilot

```bat
RUN_STAGE5_1_FULL.bat "<local_path>" "<local_path>"
```

Full يستخدم 2000 Bootstrap replicate افتراضياً. لا يُجمّد ترشيح النشر من Pilot وحده.

## مؤشرات النجاح

- `STAGE5_1_PILOT_COMPLETE.ok`
- `stage5_1_pilot_summary.json` ويجب أن يكون `status = PASS`
- `integrity_checks.csv` بلا FAIL
- `warnings_and_failures.csv` يحتوي INFO فقط
- `output_manifest_sha256.csv`

## ملاحظة منهجية

McNemar مطلوب ويُحفظ، لكنه اختبار Claim-level. بسبب ترابط Claims داخل الجلسة، يعتمد الاستدلال الأساسي على session-cluster bootstrap وعلى الاختبارات الدقيقة المقترنة على مستوى المستخدم. تحليلا Adequate/Limited وPersonal/Fallback رصديان لأن المجموعات تتكون من مستخدمين مختلفين، ولا يمثلان تجربة سببية.
