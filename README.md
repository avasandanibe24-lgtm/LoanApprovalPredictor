# Loan Approval Predictor

A classroom ML project that estimates loan approval labels and explains each prediction. The local dashboard compares three loan amounts for the same applicant.

## Run locally

Requires Python 3.10 or newer. Run `python run.py` and open http://127.0.0.1:8000. The first run trains on the Kaggle dataset. If the download fails, place loan_approval_dataset.csv from the [dataset page](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) beside run.py.

## Model results

Logistic regression powers the dashboard and scored 90.9% on 854 held-out rows. The majority-class baseline scored 62.2%. The separate offline decision-tree experiment scored 99.5% on the same split. These results describe the dataset holdout.
