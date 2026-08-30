# Can Your Phone Detect Sleep Without Listening or Tracking You?

**Chenyu Yi and Pingzhang Xu**

This project investigates whether privacy-preserving smartphone and
smartwatch measurements can distinguish between user-reported sleeping
and awake periods using the UCSD ExtraSensory dataset.

<div class="checkpoint-status" markdown="1">

**Checkpoint 2 status:** The hypothesis test and baseline model are complete.
The final model and fairness analysis are planned below and will be updated
after the remaining analysis is finished.

</div>

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
discard observations based on the response.

<div class="report-todo" markdown="1">

**Final-report item — cleaned DataFrame head:** Add a compact, privacy-reviewed
Markdown table showing the head of the cleaned DataFrame and only the columns
needed to understand the analysis.

</div>

### Univariate Analysis

The sleep response contains **202,213 not-sleeping observations**, **83,055
sleeping observations**, and **92,078 missing responses**. Not-sleeping records
are the majority, sleeping records are the minority, and about one quarter of
the response values are missing. Accuracy alone may therefore be misleading
for the later classifier.

Acceleration standard deviation is strongly right-skewed: most values are
small, with a long tail of high-motion observations. A log scale makes this
distribution easier to read without treating the extreme values as invalid.

<div class="report-todo" markdown="1">

**Required interactive figure:** Export at least one univariate Plotly figure
with `include_plotlyjs='cdn'`, place the HTML file in `assets/`, and embed it
here. The sleep-label distribution or acceleration-variability histogram from
the notebook can satisfy this requirement.

</div>

### Bivariate Analysis

The conditional sleep rate is highest from midnight through early morning and
much lower during the day, although local time does not perfectly determine
sleep. Acceleration variability also differs by reported state:

| Sleep status | Records | Mean acceleration SD | Median acceleration SD |
|---|---:|---:|---:|
| Not Sleeping | 202,200 | 0.049183 | 0.004559 |
| Sleeping | 83,054 | 0.005734 | 0.001741 |

Both the mean and median are lower during reported sleep. The distributions
still overlap, so acceleration alone cannot perfectly distinguish the two
states.

<div class="report-todo" markdown="1">

**Required interactive figure:** Export and embed at least one bivariate
Plotly figure. The hourly sleep-rate line plot or the acceleration-by-sleep
box plot from the notebook can satisfy this requirement. Keep the figure title,
axis labels, and interpretation with the plot.

</div>

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

<div class="report-todo" markdown="1">

**Required interactive figure:** Export and embed the Plotly permutation
distribution for the `is_night` test, including the observed TVD marker.

</div>

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

The final model will preserve the same training and test users. Model selection
will use group-aware cross-validation on the training users only, with the test
users evaluated once after the model and hyperparameters are fixed.

Planned improvements include:

1. Add `sin(2π × hour / 24)` and `cos(2π × hour / 24)` so the model represents
   time cyclically.
2. Add `log1p(raw_acc:magnitude_stats:std)` to reduce strong right skew while
   preserving the order of movement levels.
3. Evaluate gyroscope variability and privacy-conscious battery or screen-state
   features that are available at prediction time.
4. Compare a decision tree with a random forest.
5. Tune decision-tree `max_depth` and random-forest `n_estimators` and
   `max_depth` using `GridSearchCV` with group-aware folds.

<div class="report-todo" markdown="1">

**Complete after model selection:** State the final algorithm, every added
feature and its data-generating-process justification, the selected
hyperparameters, the tuning method, the unseen-user test performance, and the
improvement over the baseline. Optionally embed a Plotly confusion matrix or
other performance visualization.

</div>

## Fairness Analysis

The planned groups are **night observations** (Group X) and **day observations**
(Group Y) in the unchanged test set. Recall for sleeping is the evaluation
metric because failing to identify actual sleep is the error of interest.

- **Null hypothesis:** The final model is fair with respect to time period. Its
  recall for sleeping is approximately equal at night and during the day, and
  any observed difference is due to random chance.
- **Alternative hypothesis:** The final model's recall for sleeping is lower
  during the day than at night.
- **Test statistic:** Night recall minus day recall.
- **Planned method:** A one-sided permutation test at **α = 0.05** that shuffles
  the day/night group labels while keeping the fitted model, true labels, and
  predictions fixed.

<div class="report-todo" markdown="1">

**Complete after the final model is fixed:** Report both group recalls, the
observed difference, permutation p-value, decision, and conclusion. The final
model must not be retrained or modified during this analysis. An interactive
Plotly permutation distribution may also be embedded here.

</div>

---

This project was completed for DSC 80 at the University of California,
San Diego.
