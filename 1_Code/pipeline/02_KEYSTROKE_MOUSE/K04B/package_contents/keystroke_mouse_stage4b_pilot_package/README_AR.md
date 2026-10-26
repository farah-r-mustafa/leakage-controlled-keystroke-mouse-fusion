# Keystroke–Mouse Stage 4B v4.2.0

## الهدف
فتح نتائج Test بعد تجميد Stage 4A وحساب مقاييس المصادقة الكاملة دون إعادة ملاءمة المعايرة أو العتبات.

## المدخلات
- `keystroke_mouse_stage4a_review.zip`
- `keystroke_mouse_stage3b_review.zip`
- `keystroke_mouse_stage3c_review.zip`
- `keystroke_mouse_stage2a_review.zip`

## أمر التشغيل
```bat
py -3.12 "<local_path>" --input-stage4a "<local_path>" --input-stage3b "<local_path>" --input-stage3c "<local_path>" --input-stage2a "<local_path>" --output "<local_path>" --overwrite
```

## المكتبات
```bat
py -3.12 -m pip install numpy pandas scikit-learn
```

## المخرجات الرئيسة
- `overall_test_metrics.csv`
- `per_user_test_metrics.csv`
- `macro_user_test_metrics.csv`
- `model_ranking_summary.csv`
- `fallback_and_subset_sensitivity.csv`
- `tta_summary.csv`
- `tta_session_details.csv`
- `model_resource_summary.csv`
- `evaluated_test_claims/`
- `output_manifest_sha256.csv`
- `STAGE4B_COMPLETE.ok`

## تعريف TTA
الزمن من بداية أول نافذة Genuine في الجلسة إلى نهاية أول نافذة Genuine مقبولة. يعرض الزمن للجلسات الناجحة مع نسبة تغطية الجلسات بصورة منفصلة.

لا ينفذ السكربت اختبارات المقارنة الإحصائية بين النماذج؛ تؤجل إلى Stage 5.
