# Balabit Stage 5.1 Full — Unified Evaluation and Statistical Validation

هذه الحزمة تنفذ المرحلة النهائية بعد نجاح Pilot v3 وتصحيح v3.1. تعتمد على ملفات التنبؤات الكاملة الموجودة محلياً داخل:

```text
balabit_stage5_1_pilot_v3_output\full_predictions
```

ولا تنفذ أي تدريب أو استدلال جديد، ولا تعيد معايرة Temperature Scaling أو اختيار العتبات.

## الإعداد الإحصائي النهائي

- User-cluster bootstrap: **10,000** تكرار.
- Stratified session-block bootstrap: **2,000** تكرار.
- جميع النوافذ المتداخلة في الجلسة تتحرك كوحدة واحدة.
- الحفاظ على Genuine وImpostor داخل كل مستخدم وفي كل تكرار.
- Exact paired user sign-flip test مع Holm correction.
- McNemar على مستوى المطالبات محفوظ بوصفه تحليلاً وصفياً فقط.
- Seed ثابت: `2026`.

## موضع الحزمة

ضع المجلد بجانب مخرجات Pilot v3:

```text
ProjectRoot\
  balabit_stage5_1_pilot_v3_output\
    full_predictions\
  balabit_stage5_1_full_package\
```

لا تحذف مجلد `full_predictions` الموجود محلياً، حتى إن لم يكن ضمن ملف ZIP الصغير الذي رفعته للمراجعة.

## التشغيل

انقر مرتين على:

```text
run_stage5_1_full.bat
```

أو استخدم CMD:

```bat
cd /d "مسار\balabit_stage5_1_full_package"
run_stage5_1_full.bat
```

يتحقق ملف التشغيل تلقائياً من النتائج بعد اكتمال التحليل.

## المخرجات

ينشأ مجلد جديد:

```text
balabit_stage5_1_full_output\
```

ويتضمن:

- فواصل الثقة النهائية للمقاييس.
- فروق الأداء المترابطة.
- Exact sign-flip مع Holm.
- McNemar المعلّم بحدود استخدامه.
- قرار Hybrid مقابل CNN.
- جداول جاهزة للبحث بصيغتي CSV وLaTeX.
- رسوم جاهزة بصيغتي PNG بدقة 300 dpi وPDF متجه.
- تفسير بحثي باللغة الإنجليزية والعربية.
- Manifest وSHA-256 وسجل تنفيذ كامل.

## الملف المطلوب رفعه بعد التنفيذ

ارفع حصراً:

```text
balabit_stage5_1_full_output\balabit_stage5_1_full_review.zip
```

ولا ترفع مجلد `full_predictions` الكبير.

## الحماية من الكتابة فوق النتائج

إذا كان مجلد Full موجوداً من تشغيل سابق، تنقله الحزمة تلقائياً إلى اسم احتياطي مؤرخ قبل إنشاء تشغيل جديد.

## متى يتوقف التنفيذ بأمان؟

يتوقف دون تعديل المراحل السابقة إذا:

- لم يجد ملفات التنبؤات الكاملة الستة.
- لم تكن حالة Pilot v3 مكتملة.
- فشلت معايرة Validation المحفوظة أو زاد NLL بعد المعايرة.
- لم تتطابق مفاتيح السجلات والتسميات بين النماذج في البروتوكول المشترك.
- احتوى سجل التدقيق السابق على ERROR أو FATAL.

## القرار المثبت من Pilot

يظل CNN المرشح الرئيس للنشر، بينما يُعامل Hybrid كبديل موجه للدقة لا يُعتمد بوصفه متفوقاً إلا إذا دعمت نتائج Full ذلك عبر الاختبارات العنقودية، لا عبر McNemar على المطالبات وحده.
