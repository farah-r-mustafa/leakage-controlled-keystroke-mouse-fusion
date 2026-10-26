# Continuous Authentication Student Portable — Phase C+ v1.3.3

## الاسم الرسمي

**A Continuous Authentication Model in Zero Trust Architecture Using Deep Learning-Based User Behavior Analytics (UBA)**

**Student:** Farah Rashid Mustafa  
**Supervisor:** Asst. Prof. Dr.Baraa Ismeal Farhan

هذه المنصة تمثل الجزء العملي التنفيذي والتحليلي من رسالة الماجستير: تشغيل التجارب، مراقبة التنفيذ، تحليل النتائج، عرض الرسوم، وإجراء تجارب معاملات مضبوطة.

## ما الجديد في Phase C

- إضافة **The 136M Keystrokes Dataset (Aalto)** كـ external keyboard authentication benchmark.
- أصبح عدد المراحل العلمية 21 مرحلة.
- `Hyperparameter Lab` داخل صفحة Experiments.
- `Frozen Research / Baseline` يبقى مرجع إعادة الإنتاج.
- `Custom Tuning` يحفظ القيم الجديدة ولا يكتب فوق الإعدادات الأصلية.
- Snapshot من الإعدادات والأوامر يُحفظ لكل Run.
- دعم Portable Dataset Paths داخل `datasets/`.
- إضافة اسم الطالبة والمشرفة إلى Dashboard وAbout.
- Result Intelligence يقرأ نتائج Aalto الجديدة تلقائياً بعد اكتمال التشغيل.

## مدير مجموعات البيانات المتعدد Phase C+

تحتوي صفحة `Dataset & Experiment Manager` على قائمة متعددة الاختيار وأوضاع تنفيذ مضبوطة:

- Single Dataset: مجموعة واحدة.
- Independent Batch: تشغيل خطوط مستقلة تحت Run ID واحد دون خلط الصفوف أو الهويات.
- Cross-Dataset Validation: المقارنة المجمدة المدعومة بين Balabit وUEBA.
- Multimodal Fusion: متاح لمجموعتي UEBA وCyberSignature اللتين تحتويان كيبورد وماوس مقترنين.
- Transfer Learning / Harmonized Pool / Full Protocol: مسجلة للمرحلة الثانية، ويمنع تشغيلها إلى أن تثبت محولاتها بدلاً من إنتاج نتائج شكلية.

يحفظ النظام `dataset_execution_plan.json` داخل مجلد كل تشغيل لتوثيق المجموعات والنمط والتفسير العلمي.

## CyberSignature / Behaviour Biometrics Dataset

الاسمان يشيران هنا إلى مجموعة KMT نفسها. محول `CSKMT` يقرأ ZIP الأصلي مباشرة ويستخرج خصائص زمن الضغط والطيران ومسار/سرعة الماوس، ثم يقارن Keyboard وMouse وFusion ببروتوكول 5-fold out-of-fold لكل مستخدم.

ينتج كذلك Trust Score متجددًا بـEWMA وقرارات Zero Trust من ثلاث حالات: `allow` و`step_up` و`deny`، مع EER وAUC وF1 وFAR/FRR وزمن اكتشاف تبدل المستخدم. لا تُصدّر بيانات البطاقة أو أسماء المفاتيح أو الإحداثيات الخام.

## بنية البيانات المحمولة

```text
datasets/
├── Balabit/
├── Keystroke_Mouse/
│   └── Keystroke_Mouse_Dataset.zip
├── Aalto_136M/
│   └── Keystrokes.zip
└── CyberSignature_KMT/
    └── behaviour_biometrics_dataset.zip
```

عندما تكون الملفات داخل هذه البنية، تُخزن المسارات بصيغة نسبية. لذلك يمكن نسخ مجلد البرنامج بالكامل إلى حاسبة أخرى مع بقاء المسارات صالحة.

## Aalto 136M

المصدر الرسمي:
https://userinterfaces.aalto.fi/136Mkeystrokes/

التحميل:
https://userinterfaces.aalto.fi/136Mkeystrokes/data/Keystrokes.zip

DOI:
10.1145/3173574.3174220

الـPipeline الجديد:

```text
A136A  Dataset Audit & Participant Sampling
   ↓
A136B  Sequence Preparation & Leakage-Safe Section Split
   ↓
A136C  Authentication Models & Evaluation
```

A136B يستخرج dwell / down-down / up-down / keycode / rollover ويقسم على `TEST_SECTION_ID` قبل windowing.

A136C يدعم:
- Random Forest
- CNN
- BiLSTM
- CNN+BiLSTM Hybrid

ويعرض:
- Accuracy
- AUC
- EER
- FAR
- FRR
- F1
- MCC

توجد بروتوكولات Balanced 1:1 وRealistic all-impostor.

### قيد علمي مهم

Aalto 136M جُمعت في online typing test وليست مجموعة longitudinal multi-day sessions. لذلك نستخدمها **كـexternal keyboard benchmark** ولا ندمجها تلقائياً داخل Stage 5.2 الأصلي الذي قارن Balabit Mouse مع Keystroke–Mouse Mouse.

## Hyperparameter Lab

من:

`Experiments & Hyperparameter Lab → Edit Hyperparameters`

يمكن تعديل المعاملات التي تكشفها السكربتات فعلياً عبر CLI.

أمثلة Balabit Deep Models:
- epochs
- patience
- batch size
- learning rate
- weight decay
- dropout
- window size

أمثلة Aalto:
- participant limit
- sequence length / stride
- train/validation ratios
- models
- epochs / patience
- learning rate
- dropout
- hidden size
- CNN channels
- Random Forest trees

### Keystroke–Mouse K03C

السكربت المجمد الأصلي لـK03C لا يكشف معاملات deep-model training عبر CLI. لذلك يبقى Frozen في هذه النسخة، بدلاً من ادعاء دعم tuning غير موجود في الكود العلمي الأصلي.

## قاعدة المقارنة العلمية

لا تُفسر أي زيادة في رقم منفرد على أنها تحسين قبل التأكد من أن المقارنة تستخدم:
- نفس الـDataset.
- نفس المستخدمين/splits.
- نفس cohort/modality.
- نفس protocol.
- نفس metric.

مثلاً AUC لا يساوي Accuracy، وF1 لا يساوي AUC.

كما أن تقليل `participant-limit` يجعل مسألة التعرف أسهل في الغالب؛ لذلك Participant Limit هو **Benchmark Scope** وليس Hyperparameter يُخفض بهدف رفع النتيجة.

## التشغيل

1. شغّل `VERIFY_APPLICATION.bat`.
2. شغّل `START_SYSTEM.bat`.
3. ضع الـDatasets في `datasets/` أو اختر External Path مرة واحدة.
4. من Datasets اضغط `Validate & Save Portable Configuration`.
5. من Experiments اختر Dataset واحدة أو أكثر ثم اختر Execution Mode.
6. ابدأ بـFrozen Research / Baseline.
7. استخدم Custom Tuning فقط في Run ID جديد.

## اختبار Aalto الأول المقترح

للتأكد من البيئة قبل تشغيل Deep Learning:
- Participant Limit = 20 أو 30.
- Models = `random_forest`.

بعد نجاح Smoke Test:
- Participant Limit = 100 كخط أساس أولي.
- Models = `random_forest,cnn,bilstm,hybrid`.

لا تعتمد نتيجة 20/30 مشاركاً بوصفها النتيجة النهائية؛ الهدف منها فحص التشغيل فقط.


## Portable-First Fix v1.1
عند نقل الحزمة، المسارات الموجودة داخل مجلد البرنامج الحالي تتغلب تلقائياً على أي Absolute Paths قديمة محفوظة من الحاسبة السابقة. شغّل `REDETECT_PORTABLE_PATHS.bat` بعد النقل عند الحاجة، وراجع `config/PORTABLE_PATH_STATUS.json`. ملف `resolved_master_config.json` مؤقت ويعاد توليده للحاسبة الحالية.
