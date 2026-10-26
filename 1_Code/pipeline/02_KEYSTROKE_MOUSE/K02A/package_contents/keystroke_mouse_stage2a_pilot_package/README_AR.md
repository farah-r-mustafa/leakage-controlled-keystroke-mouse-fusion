# Keystroke–Mouse Stage 2A Pilot

## الغرض

تنفيذ مرحلة:

**Session Manifest, Temporal Repair and Window Feasibility Pilot**

الحزمة لا تعدل ملفات الداتاسيت الأصلية، ولا تدرب أي نموذج، ولا تنشئ Genuine/Impostor Claims. تتم القراءة صفاً بصف إلى قاعدة SQLite مؤقتة لتجنب تحميل الملف كاملاً في الذاكرة.

## بيئة التشغيل

- Windows 10
- Python 3.12+
- لا توجد مكتبات خارجية مطلوبة
- CPU فقط مناسب

## أمر التشغيل المباشر في CMD

ضع الحزمة بعد فكها مثلاً في:

`<local_path>

ثم نفّذ:

```bat
cd /d "<local_path>"
python keystroke_mouse_stage2a_pilot.py --input "<local_path>" --output "<local_path>"
```

بديل Python Launcher:

```bat
py -3.12 keystroke_mouse_stage2a_pilot.py --input "<local_path>" --output "<local_path>"
```

يجب أن يكون الأمر كله في سطر واحد. لا تكتب `-- --` قبل أي معامل.

## إعادة التنفيذ

إذا كان مجلد الإخراج موجوداً وغير فارغ، استخدم اسماً جديداً أو أضف:

```bat
--overwrite
```

## المخرجات الأساسية

- `frozen_session_manifest.csv`: التقسيم النهائي الملزم للجلسات Core.
- `all_session_chronological_candidate.csv`: التقسيم التشخيصي الأولي لجميع الجلسات، وليس التقسيم النهائي.
- `auxiliary_sessions.csv`: الجلسات المسجلة التي لم تحقق شرط Core وتبقى للتحليل المساعد.
- `primary_segment_manifest_60s.csv`: المقاطع بعد فصل فجوات الخمول الأكبر من 60 ثانية.
- `temporal_repair_audit.csv`: الانعكاسات الزمنية والإصلاح بالفرز المستقر.
- `keyboard_pairing_audit.csv`: ربط keydown/keyup.
- `mouse_stroke_audit.csv`: Mouse Strokes من الأحداث المتتالية.
- `window_configuration_summary.csv`: مقارنة بدائل النوافذ.
- `window_coverage_warnings.csv`: أي مستخدم/تقسيم بلا نوافذ في خيار معين.
- `stage2a_leakage_control_report.md`: قواعد منع التسرب الملزمة.
- `stage2a_summary.json`: الخلاصة التنفيذية.
- `output_manifest_sha256.csv`: بصمات جميع ملفات النتائج.
- ملف ZIP يحمل اسم مجلد الإخراج نفسه.

## القرارات الافتراضية المثبتة

- User1–User11 فقط هم المستخدمون المسجلون.
- `Alluserspreauth` مستبعد من الهوية الأساسية.
- Core Session تحتاج إلى 5 أزواج لوحة مفاتيح صالحة و5 Mouse Strokes صالحة على الأقل.
- الجلسات غير المؤهلة لا تحذف؛ تحفظ كـ`auxiliary_only`.
- التقسيم زمني على مستوى الجلسة بعد تطبيق شرط الأهلية الثابت.
- فصل الخمول الأساسي: 60 ثانية، مع تحليل حساسية 30 و120 ثانية.
- الحد الأعلى الأولي لـDwell Time: ثانيتان.
- لا تعبر أي نافذة حدود الجلسة الأصلية أو مقطع الخمول.

## ما يجب إرساله للمراجعة

ارفع الملف الناتج:

`keystroke_mouse_stage2a_review.zip`

ولا تبدأ Stage 2B قبل مراجعة هذه الحزمة.
