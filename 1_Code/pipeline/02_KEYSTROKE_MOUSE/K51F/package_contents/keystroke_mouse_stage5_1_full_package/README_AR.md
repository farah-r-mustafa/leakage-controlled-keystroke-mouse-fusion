# Keystroke–Mouse Stage 5.1 Full

هذه الحزمة تنفّذ التحقق الإحصائي النهائي على مخرجات Stage 4B المجمدة فقط. تشترط وجود مخرجات Pilot ناجحة ومطابقة لبصمة ملف Stage 4B.

## ما لا تفعله

- لا تعيد تدريب أي نموذج.
- لا تعيد المعايرة.
- لا تغيّر Claims أو التقسيم أو النوافذ أو العتبات.
- لا تنفّذ المقارنة مع Balabit.

## ما تنفذه

- McNemar للـClaims المقترنة.
- Session-clustered bootstrap مقترن ومطبّق داخل كل مستخدم فعلي.
- فواصل ثقة لـAUC وEER وFAR وFRR وF1 وMCC.
- Exact/Monte-Carlo paired sign-flip على مستوى المستخدم.
- Holm correction داخل العائلات وعالمياً.
- RF مقابل CNN وBiLSTM وCNN+BiLSTM.
- Keyboard وMouse وMultimodal Primary وMultimodal Any.
- Balanced وRealistic.
- Adequate/Limited وPersonal/Global fallback كتحليل رصدي.
- Scorecard نهائي يجمع الأداء والدلالة الإحصائية والكلفة الحسابية.
- ملفات موحدة جاهزة لمقارنة Balabit اللاحقة من دون تنفيذها.

## التشغيل النهائي

```bat
RUN_STAGE5_1_FULL.bat "<local_path>" "<local_path>" "<local_path>"
```

أو:

```bat
python keystroke_mouse_stage5_1_full.py ^
  --input "<local_path>" ^
  --pilot-output "<local_path>" ^
  --output "<local_path>" ^
  --bootstrap-reps 5000 ^
  --seed 5102026
```

## علامات النجاح

- `STAGE5_1_FULL_COMPLETE.ok`
- `stage5_1_full_summary.json` وفيه `status = PASS` و`mode = full`
- `integrity_checks.csv` بلا أي `FAIL`
- `deployment_recommendation.json` وفيه `status = FINAL_FULL_STAGE5_1`
- `stage5_1_full_frozen_decision.md`
- `output_manifest_sha256.csv`

## التحقق بعد التنفيذ

```bat
VERIFY_STAGE5_1_FULL_OUTPUT.bat "<local_path>"
```

لا تستخدم `--overwrite` إلا عند الرغبة الصريحة بحذف مجلد مخرجات سابق وإعادة إنشاء النتائج.
