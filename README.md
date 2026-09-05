# Stillness Has a Rhythm

### Recognizing Sleep In the Wild

**By Chenyu Yi and Pingzhang Xu**

Can an everyday phone and watch recognize sleep without listening to audio or tracking absolute
location? Using UCSD's ExtraSensory dataset, we investigated how passive motion changes during
reported sleep and built a privacy-conscious classifier evaluated on completely unseen people.

## Introduction

The ExtraSensory study followed 60 UCSD participants as they used their own phones naturally.
About once per minute, its app captured a 20-second sensor window and paired those measurements
with context labels that participants self-reported. The archive contains **377,346 one-minute
windows** across 60 anonymized users; **285,268 windows** have an observed sleeping label,
including 83,055 labeled sleeping.

Our central question is: **When participants report sleeping, how does device motion change,
and how accurately can low-privacy motion, phone-state, battery, and time features recognize
sleep for a completely unseen user?** Sleep recognition could support health monitoring, but
the in-the-wild setting is difficult: phones move independently of their owners, sensor
availability varies, and self-reports are incomplete.

Relevant columns include:

| Column | Meaning |
|---|---|
| `uuid` | Anonymized participant identifier recovered from each filename |
| `timestamp` | Unix time of the one-minute recording window |
| `label:SLEEPING` | Self-reported sleeping indicator: 1, 0, or missing |
| `raw_acc:magnitude_stats:mean` | Mean phone acceleration magnitude during the window, in G |
| `raw_acc:magnitude_stats:std` | Phone acceleration variability during the window, in G |
| `watch_acceleration:magnitude_stats:std` | Watch acceleration variability during the window, in mG |
| `discrete:app_state:*` | Binary indicators describing the phone app's state |
| `lf_measurements:battery_level` | Phone battery level |

Dataset meaning comes from the [official DSC 80 wearable instructions](https://dsc80.com/proj04/wearable/)
and the assigned [ExtraSensory paper](https://arxiv.org/pdf/1609.06354).

## Data Cleaning and Exploratory Data Analysis

We loaded the 60 compressed user files directly, selected relevant variables, and recovered each
anonymized UUID from its filename. We renamed long keys while retaining units, converted Unix
timestamps to timezone-aware San Diego local time, verified that observed sleeping labels were
only 0 or 1, and converted invalid infinities to missing values. Sensor and label gaps were kept
because missingness is part of the mobile-telemetry data-generating process.

Here is the head of the cleaned, narrow display table:

| uuid | local_time | phone_acc_mean_g | phone_acc_std_g | watch_acc_std_mg | battery_level | sleep_status |
|---|---|---:|---:|---:|---:|---|
| 00EABED2... | 2015-10-05 14:06:01-07:00 | 0.9968 | 0.0035 | 17.1951 | 0.4600 | Awake |
| 00EABED2... | 2015-10-05 14:07:01-07:00 | 0.9969 | 0.0042 | 18.9017 | 0.4600 | Awake |
| 00EABED2... | 2015-10-05 14:08:01-07:00 | 0.9968 | 0.0037 | 17.1193 | 0.4600 | Awake |
| 00EABED2... | 2015-10-05 14:09:01-07:00 | 0.9969 | 0.0035 | 17.7964 | 0.4600 | Awake |
| 00EABED2... | 2015-10-05 14:10:31-07:00 | 0.9974 | 0.0377 | 121.5919 | 0.4700 | Awake |

The univariate distribution below uses a reproducible 50,000-window display sample. Motion
variability is strongly right-skewed: phones are often almost still, with a long high-motion
tail. A logarithmic display makes both regions visible; all tests and models use the full
relevant data.

<iframe src="assets/phone-motion-distribution.html" width="100%" height="520" frameborder="0"></iframe>

The bivariate comparison shows substantially lower phone-motion variability during reported
sleep. The balanced sample is used only to keep this visualization readable.

<iframe src="assets/sleep-motion-box.html" width="100%" height="520" frameborder="0"></iframe>

Reported sleep also follows a strong daily rhythm, peaking overnight and becoming uncommon in
the afternoon.

<iframe src="assets/sleep-by-hour.html" width="100%" height="520" frameborder="0"></iframe>

This aggregate table combines time, response, and motion. Phone variability is in G and watch
variability is in mG, so the two raw magnitudes should not be compared directly.

| Local period | Labeled minutes | Sleeping (%) | Median phone motion SD (G) | Median watch motion SD (mG) |
|---|---:|---:|---:|---:|
| Night (0-5) | 66,456 | 82.6 | 0.0018 | 13.7 |
| Morning (6-11) | 66,291 | 26.0 | 0.0038 | 31.0 |
| Afternoon (12-17) | 80,922 | 3.2 | 0.0045 | 42.5 |
| Evening (18-23) | 71,599 | 11.6 | 0.0041 | 27.2 |

## Assessment of Missingness

The sleeping label is missing in 92,078 windows (24.4%). We believe it is plausibly **MNAR**:
whether a label is absent may depend on the unobserved true state because a sleeping participant
cannot answer a prompt in real time, while a busy awake participant may also ignore it. This
claim follows the label-generating process, not just a pattern in the table. Notification-open
logs, live-versus-retrospective entry flags, screen events, and research-grade actigraphy could
help explain the mechanism and make it MAR conditional on observed information.

We also ran two lecture-style permutation tests comparing categorical distributions with TVD.
Label missingness depends on local period: observed TVD **0.0713**, (p=0.0010). At
\(\alpha=0.05\), we reject independence; for example, missingness is 19.5% at night and 28.8%
in the evening.

<iframe src="assets/missingness-permutation.html" width="100%" height="520" frameborder="0"></iframe>

As a negative control, missingness does not appear related to whether a window begins in an even
or odd local minute: TVD **0.0002**, (p=0.9201). We fail to reject this second null. Dependence
on observed time is evidence consistent with MAR, but it cannot rule out the MNAR mechanism.

## Hypothesis Testing

To avoid treating repeated minutes as independent people, we first calculated each participant's
mean phone acceleration variability while awake and sleeping, then permuted the two state labels
within participants.

- **Null hypothesis:** Within a participant, awake and sleeping labels are exchangeable with
  respect to mean phone acceleration variability; the population mean awake-minus-sleep
  difference is 0.
- **Alternative hypothesis:** Mean phone acceleration variability is lower during reported
  sleep, so the awake-minus-sleep difference is positive.
- **Test statistic:** Participant-average of `mean awake phone SD - mean sleeping phone SD`, in G.
- **Significance level:** \(\alpha=0.05\).

Across 53 participants with both states, mean motion SD is 0.0502 G while awake and 0.0063 G
while sleeping. The observed participant-average difference is **0.0439 G** with
**\(p=0.00010\)**. We reject the null: there is strong evidence of lower phone-motion variability
during reported sleep. This observational association does not establish causation or prove
that a still phone means its owner is asleep.

<iframe src="assets/hypothesis-permutation.html" width="100%" height="520" frameborder="0"></iframe>

## Framing a Prediction Problem

We use **binary classification** to predict `label:SLEEPING` immediately after the same minute's
20-second sensor recording. Timestamp, motion summaries, app state, and battery level are
available then. Other self-reported labels, label-source information, future windows, audio,
and location are excluded.

The primary metric is **F1**, the harmonic mean of precision and recall. Sleep is the minority
class, so accuracy can favor the more common awake class. Precision alone ignores missed sleep;
recall alone ignores false alarms; F1 penalizes both. To test generalization without participant
leakage, `train_test_split` was applied to participant IDs: 42 users train the models and all
66,113 test windows come from 11 completely unseen users.

## Baseline Model

The baseline is one scikit-learn `Pipeline`: median imputation followed by a depth-5
`DecisionTreeClassifier`. It uses two original quantitative features—mean and standard deviation
of phone acceleration magnitude—so no categorical encoding is needed. The imputation medians
are learned from training users and reused unchanged for unseen users.

| Data | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Training users | 0.756 | 0.572 | 0.454 | 0.506 |
| Unseen test users | 0.718 | 0.585 | 0.594 | **0.590** |

This beats an always-awake rule in usefulness, but motion alone is incomplete: a sleeping
person's phone may rest on a nightstand, while an awake person's unused phone can be equally still.

## Final Model

The final single pipeline retains the baseline inputs and adds features motivated before tuning:

- sine and cosine of local hour, preserving the cyclic relationship between 11 PM and midnight;
- phone and watch interquartile motion ranges (`p75 - p25`), with watch mG converted to G;
- a watch-minus-phone motion contrast and phone coefficient of variation;
- phone/watch spectral entropy, battery level, and binary app-state indicators;
- median imputation with missingness indicators, followed by a `RandomForestClassifier`.

The feature function is inside a `FunctionTransformer`, so the same transformations are applied
during training, cross-validation, and prediction. Trees do not require standardization because
rescaling does not change the ordering used for their splits.

Before fitting, we chose `max_depth` for tuning because shallow trees can underfit and unrestricted
trees can overfit. Four participant-preserving training folds searched `[4, 6, 8, 10, 12, 16]`
with `GridSearchCV`, using F1. The search selected **depth 8** with mean validation F1 **0.794**,
then refit the 60-tree forest on all training users.

| Data | Baseline F1 | Final F1 | Absolute improvement |
|---|---:|---:|---:|
| Training users | 0.506 | 0.851 | +0.345 |
| Unseen test users | **0.590** | **0.846** | **+0.257** |

Final unseen-user accuracy is 0.903, precision is 0.918, and recall is 0.785. Similar training
and test F1 suggest depth restriction controlled variance, while the large improvement on the
unchanged test users shows that the engineered signals generalize beyond the two-feature baseline.

<iframe src="assets/model-confusion.html" width="100%" height="520" frameborder="0"></iframe>

## Fairness Analysis

Because the final model uses watch motion, we test recall parity for participants with watch
coverage below versus at/above the test-user median of 0.774.

- **Group X:** five lower-watch-coverage test participants.
- **Group Y:** six higher-watch-coverage test participants.
- **Metric:** recall, the proportion of truly sleeping windows recognized as sleeping.
- **Null hypothesis:** The model is fair with respect to watch coverage; the population recalls
  are equal and participant group labels are exchangeable.
- **Alternative hypothesis:** Recall is lower for lower-watch-coverage participants.
- **Test statistic:** `recall(higher) - recall(lower)`; large positive values favor the alternative.
- **Significance level:** \(\alpha=0.05\).

We did not refit the model. We shuffled coverage labels across the 11 test participants, keeping
each person's repeated windows together. Recall is **0.715** for lower-coverage participants and
**0.838** for higher-coverage participants, an observed gap of **0.123**. The one-sided p-value is
**0.146**, so we fail to reject the null and do not have statistically significant evidence of
worse recall for lower coverage.

<iframe src="assets/fairness-permutation.html" width="100%" height="520" frameborder="0"></iframe>

The 0.123 point estimate is still practically concerning and deserves a larger follow-up. A
non-significant result with only 11 test users is not proof of fairness. The study population is
also geographically and occupationally narrow, so other people and devices must be evaluated
before deployment. This classifier recognizes self-reported context; it is not a medical sleep
diagnosis.
