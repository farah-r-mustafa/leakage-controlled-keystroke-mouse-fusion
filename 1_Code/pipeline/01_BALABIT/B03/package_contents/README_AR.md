# Balabit Stage 3 — Genuine/Impostor Claims and Baselines

هذه المرحلة تبني مسألة **التحقق من المستخدم** بصورة صحيحة، بدلاً من تصنيف النوافذ دون تعريف مطالبة هوية.

## ما الذي تنفذه؟

لكل مستخدم مزعوم `claimed_user`:

- **Genuine claims:** نوافذ المستخدم نفسه.
- **Impostor claims:** نوافذ مستخدمين آخرين، مأخوذة بصورة متوازنة تقريباً بين المنتحلين.
- التدريب من جلسات `train` فقط.
- اختيار العتبة من `validation` فقط عند نقطة تقارب FAR وFRR.
- تطبيق العتبة المجمدة على `internal_test` فقط.

## النماذج

- Centroid Distance Baseline.
- One-Class SVM.
- Linear SVM.
- Random Forest.
- RBF-SVM متاح اختيارياً من سطر الأوامر، لكنه غير مفعل افتراضياً لكلفته العالية.

## قبل التشغيل

يجب أن يكون لديك مجلد نتائج Stage 2 الكامل، ويحتوي:

```text
window_features\
    window_features_w30.csv.gz
    window_features_w50.csv.gz
    window_features_w70.csv.gz
    window_features_w90.csv.gz
feature_dictionary.csv
protocol_manifest.csv
```

لا تختَر ملف ZIP ولا مجلد الداتاسيت الخام؛ اختر **مجلد نتائج Stage 2 نفسه**.

## ترتيب التشغيل

1. شغّل `RUN_STAGE3_PILOT.bat`.
2. ارفع `balabit_stage3_pilot_results.zip` للمراجعة.
3. بعد اعتماد النتائج شغّل `RUN_STAGE3_FULL.bat`.

## نتائج المرحلة

- `metrics_by_user.csv`: نتائج كل مستخدم وكل نموذج.
- `metrics_macro.csv`: المتوسط والانحراف والمدى عبر المستخدمين.
- `metrics_pooled.csv`: النتائج المجمعة باستخدام عتبة كل مستخدم.
- `thresholds.csv`: العتبات المختارة من التحقق.
- `claim_counts.csv`: أعداد Genuine/Impostor لكل تقسيم.
- `protocol_audit.csv`: تدقيق تسرب الجلسات وبنية التقسيم.
- `window_model_ranking.csv`: ترتيب أولي لأفضل حجم نافذة ونموذج.
- `test_predictions.csv.gz`: تنبؤات الاختبار لاستخدامها لاحقاً في McNemar والتحليل الإحصائي.
- `stage3_summary.json` و`run_log.txt`.

## ملاحظة منهجية

EER في `eer_test_curve` يحسب من منحنى درجات الاختبار لأغراض المقارنة، بينما قرارات Accuracy/FAR/FRR/F1/MCC تستخدم حصراً **العتبة التي اختيرت على Validation**.
