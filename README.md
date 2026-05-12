# KKBox Subscription Churn Prediction

This repository is a project for music streaming subscription churn modeling using
the WSDM-KKBox Churn Prediction Challenge data.

The task is to predict whether a user will churn after the current subscription expires.
For KKBox's 30-day subscription model, churn is defined as having no new valid service
subscription within 30 days after the current membership expiration date.

Key transaction fields used to determine renewal or churn include `transaction_date`,
`membership_expire_date`, and `is_cancel`. A cancellation event does not always mean
the user has churned, because users may cancel, change plans, or resubscribe before the
30-day grace window closes.

The train and test sets are split by transaction timing. In the competition description,
the training set contains users whose subscriptions expire in **February 2017**, so their
renewal or churn outcome is observed around March 2017. The refreshed test data targets
users whose subscriptions expire in **March 2017**, with churn or renewal observed around
April 2017. Additional user behavior data is included beyond the train and test label
files so participants can build features from listening activity, transaction history,
and member profile signals.

## Dataset

This project uses the full Kaggle competition dataset.

### Tables

- `train.csv`: Training labels with `msno` and `is_churn`.
- `train_v2.csv`: Same format as `train.csv`; refreshed on 2017-11-06 and contains
  March 2017 churn labels.
- `sample_submission_zero.csv`: Test user IDs in the required submission format,
  with `msno` and the `is_churn` value to predict.
- `sample_submission_v2.csv`: Same format as `sample_submission_zero.csv`; refreshed
  on 2017-11-06 and contains the April 2017 test set.
- `transactions.csv`: User transaction history through 2017-02-28.
- `transactions_v2.csv`: Same format as `transactions.csv`; refreshed on 2017-11-06
  and contains transaction history through 2017-03-31.
- `user_logs.csv`: Daily user listening behavior through 2017-02-28.
- `user_logs_v2.csv`: Same format as `user_logs.csv`; refreshed on 2017-11-06 and
  contains listening logs through 2017-03-31.
- `members.csv`: User profile snapshot. Not every user has member metadata.
- `members_v3.csv`: Refreshed on 2017-11-13 and replaces `members.csv`; the
  snapshot `expiration_date` field was removed.

### Important Fields

Label files:

- `msno`: Anonymous user ID.
- `is_churn`: Target variable. `1` means the user did not continue the subscription
  within 30 days after expiration; `0` means renewal.

Transaction files:

- `payment_method_id`: Payment method.
- `payment_plan_days`: Membership plan length in days.
- `plan_list_price`: Listed plan price in New Taiwan Dollar (NTD).
- `actual_amount_paid`: Actual amount paid in New Taiwan Dollar (NTD).
- `is_auto_renew`: Whether the plan is set to auto-renew.
- `transaction_date`: Transaction date in `%Y%m%d` format.
- `membership_expire_date`: Membership expiration date in `%Y%m%d` format.
- `is_cancel`: Whether the user canceled the membership in this transaction.

User log files:

- `date`: Listening log date in `%Y%m%d` format.
- `num_25`, `num_50`, `num_75`, `num_985`, `num_100`: Counts of songs played up to
  different completion thresholds.
- `num_unq`: Number of unique songs played.
- `total_secs`: Total seconds played.

Member files:

- `city`: User city code.
- `bd`: Age. This field contains outliers, including negative and unrealistic values.
- `gender`: User gender when available.
- `registered_via`: Registration method.
- `registration_init_time`: Registration date in `%Y%m%d` format.
- `expiration_date`: Snapshot expiration date in `members.csv`; removed from
  `members_v3.csv` because it does not represent actual churn behavior.

### Data Extraction Details

The competition provides `WSDMChurnLabeller.scala` to generate labels for users in the
prediction scope. The original cluster log history spans 2015-01-01 to 2017-03-31.
With this label generator, participants can create additional training labels beyond
the sample labels provided by the competition.

The key extraction detail is the definition of the current membership expiration date.
A user is in scope when the relevant expiration date falls inside the target prediction
month. The user is labeled as churned only if there is no valid new subscription within
30 days after that expiration date.

Active cancellation can move the current expiration date earlier, so `is_cancel = 1`
does not automatically mean churn. For example, if cancellation moves the expiration
date to 2017-03-16 and the user buys another plan on 2017-04-01, the renewal happens
within 30 days and the user is not churned. If a later transaction extends the
membership expiration date outside the target month, the user is not included in that
prediction window.
