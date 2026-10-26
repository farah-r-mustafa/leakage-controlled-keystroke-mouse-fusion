# Keystroke–Mouse Stage 4A v4.1.0

هذه المرحلة تنفذ المعايرة واختيار العتبات باستخدام **Balanced Validation فقط**، ثم تطبق السياسة المجمدة على Validation وTest على مستوى الـClaim. لا تحسب أي مقاييس مجمعة على Test.

## التشغيل

```bat
py -3.12 "<local_path>" --input-stage3b "<local_path>" --input-stage3c "<local_path>" --output "<local_path>" --overwrite
```

## السياسة

- Temperature Scaling: `sigmoid(raw_score / T)`.
- بيانات الملاءمة: Balanced Validation فقط.
- العتبة: نقطة EER من Validation فقط.
- المعايرة/العتبة الشخصية تتطلب 20 Genuine و20 Impostor وخمس هويات Impostor على الأقل.
- عند عدم الكفاية يستخدم Global fallback موزوناً بالتساوي بين المستخدمين والفئتين.
- لا يعكس السكربت اتجاه الدرجة بناءً على Validation.
- لا يحسب Test FAR/FRR/EER/AUC أو أي نتيجة نهائية.
