# Balabit Stage 4 — بناء المصفوفات التسلسلية

هذه المرحلة تبني مدخلات صحيحة لـ:

- CNN
- BiLSTM
- CNN+BiLSTM

بدلاً من تمرير الميزات الإحصائية المسطحة إلى نموذج زمني.

## ما تحافظ عليه المرحلة؟

- تستخدم `protocol_manifest.csv` الناتج من Stage 2.
- لا تنشئ تقسيماً جديداً.
- لا تسمح بعبور النافذة حدود الجلسة.
- تضبط القص والتقييس من أحداث Train فقط.
- لا تستخدم Validation أو Internal Test في إعداد المعالجة.
- تستبعد `fake_data` والاختبار الرسمي غير الموسوم.
- لا تدخل `user_id` أو `session_id` أو الوقت المطلق ضمن ميزات النموذج.

## الميزات على مستوى كل حدث

### مستمرة ومقيسة من التدريب فقط

- dt
- dx, dy
- distance
- speed
- acceleration
- jerk
- angular_velocity
- x, y
- sin(angle), cos(angle)

### ثنائية غير مقيسة

- Move, Drag, Pressed, Released
- No button, Left, Right, Middle
- Idle gap

المجموع: **21 ميزة لكل حدث**.

---

# قبل التشغيل

يجب أن يتوفر لديك:

1. مجلد بيانات Balabit الخام الذي يحتوي `training_files`.
2. مجلد نتائج Stage 2 الكامل الذي يحتوي:
   `protocol_manifest.csv`

لا تختَر ملف ZIP نفسه؛ فك ضغطه أو استخدم مجلد النتائج المحلي.

---

# التشغيل التجريبي أولاً

شغّل:

```text
RUN_STAGE4_PILOT.bat
```

سيبني:

- المستخدمين: user12, user15, user16
- النوافذ: 30 و70

الناتج الافتراضي:

```text
balabit_stage4_pilot
balabit_stage4_pilot_review.zip
```

ارفع ملف:

```text
balabit_stage4_pilot_review.zip
```

ولا ترفع مجلد `tensors` لأنه كبير.

---

# التشغيل الكامل

بعد مراجعة الـPilot شغّل:

```text
RUN_STAGE4_FULL.bat
```

وسيستخدم:

- المستخدمين العشرة
- النوافذ 30 و50 و70 و90

الناتج الافتراضي:

```text
balabit_stage4_full
balabit_stage4_full_review.zip
```

يجب إبقاء مجلد `balabit_stage4_full` محلياً، لأن Stage 5 سيقرأ ملفات `.npy` الموجودة داخله.

---

# المخرجات المهمة

```text
event_preprocessor.npz
feature_schema.csv
users.csv
sessions.csv
tensor_counts.csv
tensor_manifest.csv
protocol_audit.csv
stage4_summary.json
stage4_errors.csv
session_quality_stage4.csv
run_log.txt
tensors\
```

داخل كل حجم نافذة:

```text
tensors\w30\
  train_X.npy
  train_y_user.npy
  train_session_index.npy
  train_window_index.npy
  train_metadata.csv.gz
  validation_X.npy
  ...
  internal_test_X.npy
  ...
```

---

# ملاحظات الموارد

- التنفيذ الكامل قد ينتج أكثر من 1 GB تقريباً حسب أحجام النوافذ.
- لا تنقل ملفات `.npy` أثناء التشغيل.
- تأكد من وجود مساحة حرة كافية.
- التقييس يحدث مرة واحدة، ولا تحتاج Stage 5 إلى إعادة بنائه.
