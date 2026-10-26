# Balabit Stage 5 — CNN وBiLSTM والنموذج الهجين

تدرب هذه الحزمة ثلاثة نماذج على المصفوفات التسلسلية الناتجة من Stage 4:

1. CNN
2. BiLSTM
3. CNN+BiLSTM

## بروتوكول المصادقة

يُدرّب كل نموذج أولاً كمصنف مشترك للمستخدمين العشرة. بعد ذلك:

- احتمال المستخدم المطلوب يمثل درجة الثقة للمطالبة.
- نوافذ المستخدم نفسه تمثل Genuine Claims.
- نوافذ المستخدمين الآخرين تمثل Impostor Claims.
- تُختار عتبة منفصلة لكل مستخدم من Validation فقط.
- تُجمّد العتبات قبل اختبار Internal Test.
- تُحسب FAR وFRR وEER وAUC وF1 وMCC.
- تُقاس Latency وTTA.
- تكون نتائج Macro per-user هي النتائج الأساسية.

لا تعيد هذه المرحلة تقسيم البيانات ولا تعيد ضبط المعالجة المسبقة.

---

## المتطلبات

يجب الاحتفاظ بمجلد Stage 4 الكامل، وبداخله مثلاً:

```text
balabit_stage4_full\
  users.csv
  sessions.csv
  tensors\
    w50\
      train_X.npy
      train_y_user.npy
      train_session_index.npy
      train_metadata.csv.gz
      validation_X.npy
      ...
      internal_test_X.npy
      ...
```

---

## تشغيل الـPilot

شغّل:

```text
RUN_STAGE5_PILOT_W50.bat
```

واختر مجلد:

```text
balabit_stage4_full
```

وليس مجلد `tensors` أو `w50` أو ملف ZIP.

إعدادات الـPilot:

- Window = 50
- المستخدمون العشرة
- حتى 2,500 نافذة تدريب لكل مستخدم
- حتى 800 نافذة Validation لكل مستخدم
- حتى 1,000 نافذة Test لكل مستخدم
- 15 Epochs كحد أقصى
- Early stopping
- Batch size = 128
- نفس العينة المحددة للنماذج الثلاثة

الناتج المطلوب رفعه:

```text
balabit_stage5_pilot_w50_review.zip
```

احتفظ بالمجلد الكامل محلياً، لأنه يحتوي ملفات النماذج والتنبؤات.

---

## التشغيل الكامل لنافذة 50

لا تشغّل هذا الملف قبل مراجعة نتائج الـPilot:

```text
RUN_STAGE5_FULL_W50.bat
```

الإعداد الكامل يستخدم جميع نوافذ Window 50، مع 30 Epochs كحد أقصى.

---

## الملفات المهمة

```text
stage5_summary.json
training_history.csv
metrics_by_user.csv
metrics_macro.csv
metrics_pooled.csv
thresholds.csv
latency_tta.csv
multiclass_metrics.csv
confusion_matrices.csv
model_inventory.csv
selection_summary.csv
protocol_audit.csv
stage5_errors.csv
run_log.txt
models\
predictions\
```

ملف المراجعة الصغير لا يحتوي النماذج أو التنبؤات الكبيرة.

---

## ملاحظة مهمة

نتائج Random Forest وLinear SVM في Stage 3 مبنية على ميزات إحصائية ثابتة، بينما هذه النماذج تتلقى تسلسلات الأحداث المرتبة. المقارنة النهائية تكون على:

- التقسيم نفسه
- أحجام النوافذ نفسها
- المستخدمين أنفسهم
- مقاييس المصادقة نفسها

ولا تُحسم أفضلية النموذج من Accuracy وحدها؛ تُعتمد EER وFAR وMCC وTTA أيضاً.
