# Can Your Phone Detect Sleep Without Listening or Tracking You?

**Chenyu Yi and Pingzhang Xu**

This project investigates whether privacy-preserving smartphone and
smartwatch measurements can distinguish between user-reported sleeping
and awake periods using the UCSD ExtraSensory dataset.

The full analysis is complete. The final random-forest model is evaluated on
participants who were not seen during training, followed by a permutation-based
fairness analysis comparing daytime and nighttime sleep recognition.

<nav class="report-nav" aria-label="Report sections">
  <a href="#introduction">Introduction</a>
  <a href="#data-cleaning-and-exploratory-data-analysis">Cleaning and EDA</a>
  <a href="#assessment-of-missingness">Missingness</a>
  <a href="#hypothesis-testing">Hypothesis Testing</a>
  <a href="#framing-a-prediction-problem">Prediction Problem</a>
  <a href="#baseline-model">Baseline Model</a>
  <a href="#final-model">Final Model</a>
  <a href="#fairness-analysis">Fairness Analysis</a>
</nav>

## Introduction

The ExtraSensory dataset contains smartphone and smartwatch measurements
collected from 60 people during everyday life. Approximately 20 seconds of
sensor readings were summarized in each roughly one-minute observation. The
measurements include motion, location, audio, phone state, battery level,
screen brightness, and context labels reported by participants.

The prediction design intentionally excludes audio and location measurements.
It focuses on motion, time, and phone-state summaries that are less revealing
about the content of a person's activities or conversations.

This project asks:

> **How does passive smartphone activity differ when a user is sleeping
> versus not sleeping, and can ordinary phone measurements predict whether
> the user is sleeping?**

Sleep recognition may reduce the burden of manual self-reporting and support
wellness applications. A useful system should also respect privacy, handle
missing self-reports, and generalize to people who were not represented during
training.

After the 60 participant files were combined, the working dataset contained
**377,346 rows**. Each row represents one sensor-summary window for one
anonymized user. The columns most relevant to this investigation are:

| Column | Meaning | Role |
|---|---|---|
| `uuid` | Anonymized user identifier | Grouping and user-level splitting |
| `timestamp` | Unix time for the observation | Source for local-time features |
| `label:SLEEPING` | 1 for sleeping, 0 for not sleeping, or missing | Response variable |
| `raw_acc:magnitude_stats:std` | Standard deviation of phone acceleration magnitude | Motion feature |
| `discrete:app_state:is_active` | Whether an app was active | Phone-state feature |
| `lf_measurements:battery_level` | Phone battery level | Candidate phone-state feature |
| `lf_measurements:screen_brightness` | Screen brightness | Candidate phone-state feature |
| `hour` | Local hour derived during cleaning | Time feature |
| `is_night` | Whether local time is from 9 PM through 5:59 AM | Time-period feature |

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

The 60 compressed participant files were combined row-wise after adding an
anonymized user identifier derived from each filename. The following cleaning
steps were then applied:

1. Convert Unix timestamps to San Diego local time and derive `hour`.
2. Mark observations from 9 PM through 5:59 AM with `is_night = 1`.
3. Verify that every observed `label:SLEEPING` value is either 0 or 1.
4. Remove duplicate user–timestamp rows; no duplicate pairs remained.
5. Remove `location:max_speed`, which was more than 70% missing and was not
   needed for the sleep question.
6. Retain missing sleep labels for the missingness analysis rather than
   replacing them with invented outcomes.

The cleaned dataset still contains **377,346 rows**. Cleaning changed the
representation of time and removed an irrelevant sparse column, but it did not
discard observations based on the response. A compact view of the cleaned data
is shown below; user identifiers remain anonymized.

| User | Local datetime | Sleeping | Acceleration SD | App active | Hour | Night |
|---|---|---:|---:|---:|---:|---:|
| No1 | 2015-11-05 14:24:57-08:00 | 0 | 0.040597 | 0 | 14 | 0 |
| No1 | 2015-11-05 14:25:57-08:00 | 0 | 0.006165 | 0 | 14 | 0 |
| No1 | 2015-11-05 14:26:57-08:00 | 0 | 0.006302 | 0 | 14 | 0 |
| No1 | 2015-11-05 14:28:07-08:00 | 0 | 0.004767 | 0 | 14 | 0 |
| No1 | 2015-11-05 14:29:08-08:00 | 0 | 0.005415 | 0 | 14 | 0 |

### Univariate Analysis

<iframe
  src="assets/sleep_distribution.html"
  width="100%"
  height="500"
  frameborder="0">
</iframe>

The sleep response contains **202,213 not-sleeping observations**, **83,055
sleeping observations**, and **92,078 missing responses**. Not-sleeping records
are the majority, sleeping records are the minority, and about one quarter of
the response values are missing. Accuracy alone may therefore be misleading
for the later classifier.

Acceleration standard deviation is strongly right-skewed: most values are
small, with a long tail of high-motion observations. A log scale makes this
distribution easier to read without treating the extreme values as invalid.

<iframe
  src="assets/acceleration_distribution.html"
  width="100%"
  height="500"
  frameborder="0">
</iframe>

### Bivariate Analysis

The conditional sleep rate is highest from midnight through early morning and
much lower during the day, although local time does not perfectly determine
sleep. Acceleration variability also differs by reported state:

<iframe
  src="assets/hourly_sleep_rate.html"
  width="100%"
  height="500"
  frameborder="0">
</iframe>

| Sleep status | Records | Mean acceleration SD | Median acceleration SD |
|---|---:|---:|---:|
| Not Sleeping | 202,200 | 0.049183 | 0.004559 |
| Sleeping | 83,054 | 0.005734 | 0.001741 |


Both the mean and median are lower during reported sleep. The distributions
still overlap, so acceleration alone cannot perfectly distinguish the two
states.

<iframe
  src="assets/acceleration_by_sleep.html"
  width="100%"
  height="550"
  frameborder="0">
</iframe>


### Interesting Aggregates

Grouping by night status and app activity produces the following conditional
sleep rates:

| Period | App state | Sleep rate | Records |
|---|---|---:|---:|
| Day | Active | 0.016 | 4,669 |
| Day | Not Active | 0.118 | 180,899 |
| Night | Active | 0.174 | 929 |
| Night | Not Active | 0.623 | 98,771 |

Nighttime records have a much higher sleep rate, and active app use is
associated with a lower sleep rate. These are observational associations, not
evidence that either condition causes sleep.

## Assessment of Missingness

### MNAR Analysis

`label:SLEEPING` is plausibly **MNAR** because its missingness may depend on the
unobserved true value. A participant who is asleep cannot report that state at
the time, and a tired participant may ignore a prompt. This assessment follows
from the data-generating process rather than from the observed DataFrame alone.
Prompt times, response delays, charging state, and whether the screen was on
could help explain the missingness and make an MAR explanation more plausible.

### Missingness Dependency

For each observed binary feature, the test statistic is the absolute difference
in the proportion equal to 1 between rows where `label:SLEEPING` is missing and
rows where it is observed. For a binary distribution, this statistic is equal
to total variation distance (TVD). Each test uses 2,000 permutations and
**α = 0.001**.

| Feature | Observed TVD | p-value | Decision at α = 0.001 |
|---|---:|---:|---|
| `is_night` | 0.022491 | 0.000500 | Reject the null |
| `discrete:time_of_day:between21and3` | 0.005555 | 0.001499 | Fail to reject the null |

The test provides evidence that sleep-label missingness depends on the broader
night period. It does not provide sufficient evidence of dependence on the
narrower 9 PM–3 AM indicator. Failing to reject the second null does not prove
independence or establish that the response is MCAR.

<iframe
  src="assets/missingness_permutation.html"
  width="100%"
  height="500"
  frameborder="0">
</iframe>


## Hypothesis Testing

The project question suggests testing whether the phone moves less during
reported sleep.

- **Null hypothesis:** In the population represented by these observations,
  mean phone acceleration standard deviation is the same while users are
  sleeping and not sleeping. Any observed difference is due to random chance
  under the null.
- **Alternative hypothesis:** Mean phone acceleration standard deviation is
  lower while users are sleeping.
- **Test statistic:** Mean acceleration standard deviation for not-sleeping
  observations minus the mean for sleeping observations. Large positive values
  favor the alternative.
- **Method:** One-sided permutation test with 1,000 repetitions.
- **Significance level:** **α = 0.01**.

The observed test statistic was **0.043449**, and the permutation p-value was
**0.000999**. Because the p-value is below 0.01, we reject the null hypothesis.
The observed data provide strong evidence that mean phone acceleration
variability is lower during reported sleep. Because the data are observational,
this result identifies an association and does not establish that sleep causes
the difference.

<iframe
  src="assets/hypothesis_test.html"
  width="100%"
  height="500"
  frameborder="0">
</iframe>

## Framing a Prediction Problem

The prediction task is to predict `label:SLEEPING` from passive measurements
available in the same one-minute window. This is **binary classification**, with
sleeping as the positive class. Rows with a missing response are excluded from
modeling because their correct class is unknown.

The primary evaluation metric is the **F1-score for sleeping**. Sleeping is the
minority class, so accuracy can be high even when many sleeping observations
are missed. F1-score balances precision and recall, while accuracy and the full
classification report serve as secondary diagnostics.

Local time, acceleration variability, and current app state are available at
the time of prediction. Other self-reported context labels are excluded because
they may not be available and could leak information about the response. Users,
rather than individual rows, are split between the training and test sets so
that test performance measures generalization to unseen people.

## Baseline Model

A fixed time-only benchmark predicts sleeping from 9 PM through 5:59 AM and not
sleeping at all other times. The required baseline is a class-balanced,
depth-3 decision tree implemented with its preprocessing in a single
`sklearn` Pipeline.

| Feature | Type | Processing |
|---|---|---|
| `hour` | Quantitative | Missing values replaced with training-set median |
| `raw_acc:magnitude_stats:std` | Quantitative | Missing values replaced with training-set median |
| `discrete:app_state:is_active` | Binary nominal | Missing values replaced with training-set median |

No one-hot encoding is needed because the selected columns are numeric or
binary. The imputer is fitted only on the training data. The split contains
231,018 rows from 42 training users and 54,250 rows from 11 unseen test users.

| Model and split | Accuracy | Precision for sleeping | Recall for sleeping | F1 for sleeping |
|---|---:|---:|---:|---:|
| Time-only benchmark, test | 0.766 | 0.528 | 0.742 | 0.617 |
| Three-feature baseline, train | 0.879 | — | — | 0.810 |
| Three-feature baseline, unseen-user test | 0.864 | 0.695 | 0.827 | 0.755 |

The baseline substantially improves on the time-only benchmark, which shows
that motion and app activity add useful predictive information. It is a useful
starting model, but it is not yet the final model: its test F1-score is lower
than its training F1-score, numeric hour does not capture the circular
relationship between 11 PM and midnight, and only one motion summary is used.

## Final Model

The final model preserves exactly the same training and test users as the
baseline. It is a **random forest with 120 trees**, selected because it can
capture nonlinear interactions among time, movement, and phone state while
averaging many decision trees for more stable predictions.

Four features were engineered for reasons tied to the data-generating process:

1. `hour_sin` and `hour_cos` represent the 24-hour cycle, so 11 PM and midnight
   are close rather than opposite ends of a numeric scale.
2. `log_acc_std` reduces the strong right skew in acceleration variability and
   preserves distinctions among low-motion observations relevant to sleep.
3. `log_gyro_std` supplies a second measure of phone movement and addresses its
   right-skewed distribution.

App-active state, battery level, and screen brightness are retained as
privacy-conscious indicators of phone use. Feature engineering, median
imputation, and classification are contained in a single sklearn Pipeline.

Before tuning, we chose to search `max_depth`, which controls tree complexity,
and `min_samples_leaf`, which prevents rules based on very small groups of
records. `GridSearchCV` used three-fold `GroupKFold` cross-validation on the
training users and sleeping-class F1 as its scoring metric. The best settings
were **`max_depth=8`** and **`min_samples_leaf=50`**, with mean validation F1 of
**0.816**.

| Model | Accuracy | Precision for sleeping | Recall for sleeping | F1 for sleeping |
|---|---:|---:|---:|---:|
| Baseline decision tree | 0.864 | 0.695 | 0.827 | 0.755 |
| Final random forest | **0.874** | **0.715** | **0.836** | **0.771** |

The final model improves the primary metric from 0.755 to 0.771 on the unchanged
unseen-user test set. Precision, recall, and accuracy also improve.

<iframe
  src="assets/final_confusion_matrix.html"
  width="100%"
  height="520"
  frameborder="0">
</iframe>

## Fairness Analysis

We test whether the final model performs worse at recognizing actual sleep
during the **day** than during the **night** in the unchanged test set. Night is
defined as 9 PM through 5:59 AM. This is meaningful because a model that mainly
learns conventional nighttime schedules may miss naps or sleep among users with
nontraditional schedules. Recall for sleeping is the evaluation metric because
failing to identify actual sleep is the error of interest.

- **Null hypothesis:** The final model's recall for sleeping is the same during
  night and day; the observed difference is due to random assignment of the
  time-period labels.
- **Alternative hypothesis:** The final model's recall for sleeping is lower
  during the day than at night.
- **Test statistic:** Night recall minus day recall.
- **Method:** A one-sided permutation test with 5,000 repetitions at
  **α = 0.05** that shuffles
  the day/night group labels while keeping the fitted model, true labels, and
  predictions fixed.

The model's sleeping-class recall is **0.907 at night** and **0.632 during the
day**, giving an observed night-minus-day difference of **0.276**. The one-sided
permutation p-value is approximately **0.0002**. Because this is below 0.05, we
reject the null hypothesis. The data provide strong evidence that this model
performs worse at recognizing actual daytime sleep than nighttime sleep. This
identifies a specific recall disparity for these time-period groups; it does not
establish that the model is unfair in every possible sense.

<iframe
  src="assets/fairness_permutation.html"
  width="100%"
  height="500"
  frameborder="0">
</iframe>

---

This project was completed for DSC 80 at the University of California,
San Diego.
