"""Train and evaluate an educational loan approval classifier."""
import argparse
import io
from zipfile import ZipFile, BadZipFile
import csv
import json
import math
import random
import statistics
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/architsharma01/loan-approval-prediction-dataset"
FEATURES = ['income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'no_of_dependents', 'education', 'self_employed', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
NUMERIC = set(FEATURES) - {"education", "self_employed"}
DEFAULT_DATA = ROOT / "loan_approval_dataset.csv"
MODEL_PATH = ROOT / "model.json"
METRICS_PATH = ROOT / "metrics.json"


def download_data(path):
    try:
        with urlopen(KAGGLE_URL, timeout=30) as response:
            payload = response.read()
        with ZipFile(io.BytesIO(payload)) as archive:
            match = next(name for name in archive.namelist() if Path(name).name == path.name)
            path.write_bytes(archive.read(match))
    except (OSError, BadZipFile, StopIteration) as error:
        raise ValueError("Download failed. Download the Kaggle CSV using the README link.") from error


def read_data(path):
    if path == DEFAULT_DATA and not path.exists():
        download_data(path)
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        columns = {name.strip().lower() for name in reader.fieldnames or []}
        missing = set(FEATURES + ["loan_status"]) - columns
        if missing:
            raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
        rows = set()
        original_count = 0
        for line, record in enumerate(reader, 2):
            if None in record:
                raise ValueError(f"Extra CSV fields on line {line}.")
            row = {key.strip().lower(): value for key, value in record.items()}
            label = (row["loan_status"] or "").strip().lower()
            if label not in {"approved", "rejected"}:
                raise ValueError(f"Invalid loan_status on line {line}.")
            values = normalize_inputs(row)
            rows.add((tuple(values[name] for name in FEATURES), int(label == "approved")))
            original_count += 1
    return sorted(rows, key=repr), original_count


def normalize_inputs(values):
    # Normalize equivalent numbers and missing markers before detecting duplicates.
    normalized = {}
    for name in FEATURES:
        value = values[name]
        if value is None or str(value).strip() in {"", "?"}:
            normalized[name] = None
        elif name in NUMERIC:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{name} must be a finite number.")
            normalized[name] = number
        else:
            normalized[name] = str(value).strip().casefold()
    return normalized


def split_rows(rows, seed=42):
    # Keep identical feature profiles together across the train/test boundary.
    by_profile = {}
    for values, label in rows:
        by_profile.setdefault(values, []).append((values, label))
    rng = random.Random(seed)
    train, test = [], []
    for labels in [(0,), (1,), (0, 1)]:
        group = [items for items in by_profile.values()
                 if tuple(sorted(label for _, label in items)) == labels]
        rng.shuffle(group)
        count = min(len(group) - 1, max(1, round(len(group) * 0.2))) if group else 0
        for items in group[:count]:
            test.extend(items)
        for items in group[count:]:
            train.extend(items)
    if any({label for _, label in part} != {0, 1} for part in (train, test)):
        raise ValueError("Both splits need positive and negative examples. Supply more distinct profiles.")
    return train, test


def fit_encoder(rows):
    # Learn medians, scaling statistics, and category vocabularies from training only.
    encoder = {}
    for name, column in zip(FEATURES, zip(*(values for values, _ in rows))):
        known = [value for value in column if value is not None]
        if name in NUMERIC:
            median = statistics.median(known) if known else 0.0
            filled = [value if value is not None else median for value in column]
            encoder[name] = {
                "median": median,
                "mean": statistics.mean(filled),
                "scale": statistics.pstdev(filled) or 1.0,
            }
        else:
            encoder[name] = {"categories": sorted(set(known))}
    return encoder


def encode(values, encoder):
    features = [1.0]  # The intercept lets the model learn a base rate.
    for name, value in zip(FEATURES, values):
        rules = encoder[name]
        if name in NUMERIC:
            filled = rules["median"] if value is None else value
            features.append((filled - rules["mean"]) / rules["scale"])
        else:
            categories = rules["categories"]
            features.extend(float(value == category) for category in categories)
            features.append(float(value not in categories))
    return features


def sigmoid(score):
    # Evaluate the sigmoid without overflowing for large negative scores.
    if score >= 0:
        return 1 / (1 + math.exp(-score))
    exponential = math.exp(score)
    return exponential / (1 + exponential)


def predict(encoded, weights):
    return sigmoid(sum(value * weight for value, weight in zip(encoded, weights)))


def fit_logistic(rows, encoder, steps=800, rate=0.15, penalty=0.01):
    encoded_rows = [(encode(values, encoder), label) for values, label in rows]
    weights = [0.0] * len(encoded_rows[0][0])
    for _ in range(steps):
        gradient = [0.0] * len(weights)
        for features, label in encoded_rows:
            error = predict(features, weights) - label
            for index, value in enumerate(features):
                gradient[index] += error * value
        for index, slope in enumerate(gradient):
            regularization = penalty * weights[index] if index else 0.0
            weights[index] -= rate * (slope / len(rows) + regularization)
    return weights


def evaluate(labels, scores):
    # Confusion matrix rows are actual classes, columns are predictions.
    matrix = [0, 0, 0, 0]
    for label, score in zip(labels, scores):
        matrix[2 * label + int(score >= 0.5)] += 1
    tn, fp, fn, tp = matrix
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    # ROC AUC is the fraction of positive/negative pairs correctly ranked,
    # counting ties as half a correct pair.
    auc = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return {
        "accuracy": (tp + tn) / len(labels),
        "precision_approved": tp / (tp + fp) if tp + fp else 0.0,
        "recall_approved": tp / (tp + fn),
        "f1_approved": 2 * tp / (2 * tp + fp + fn),
        "roc_auc": auc / (len(positives) * len(negatives)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def train(data_path, model_path, metrics_path):
    rows, original_count = read_data(data_path)
    train_rows, test_rows = split_rows(rows)
    encoder = fit_encoder(train_rows)
    weights = fit_logistic(train_rows, encoder)
    actual = [label for _, label in test_rows]
    scores = [predict(encode(values, encoder), weights) for values, _ in test_rows]
    base_score = statistics.mean(label for _, label in train_rows)
    metrics = {
        "data_file": data_path.name,
        "raw_rows": original_count,
        "duplicates_removed": original_count - len(rows),
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "approved_rows": sum(label for _, label in rows),
        "rows_with_missing_features": sum(None in values for values, _ in rows),
        "basis": "Fixed 0.5 threshold and held-out, approximately 20% test split",
        "majority_baseline": evaluate(actual, [base_score] * len(actual)),
        "logistic_regression": evaluate(actual, scores),
    }
    # Record the financial ranges observed during training to avoid extrapolation.
    financial_ranges = {}
    for name in ("income_annum", "loan_amount"):
        index = FEATURES.index(name)
        observed = [values[index] for values, _ in train_rows if values[index] is not None]
        if not observed:
            raise ValueError(f"Training data has no values for {name}.")
        financial_ranges[name] = [min(observed), max(observed)]
    income_index = FEATURES.index("income_annum")
    amount_index = FEATURES.index("loan_amount")
    ratios = [values[amount_index] / values[income_index] for values, _ in train_rows
              if values[income_index] and values[income_index] > 0
              and values[amount_index] is not None]
    if not ratios:
        raise ValueError("Training data needs examples with both income and loan amount.")
    financial_ranges["max_loan_to_income"] = max(ratios)
    model = {"encoder": encoder, "weights": weights, "financial_ranges": financial_ranges}
    model_path.write_text(json.dumps(model, indent=2, allow_nan=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    return metrics


def predict_application(input_path, model_path):
    application = json.loads(input_path.read_text(encoding="utf-8-sig"))
    return score_application(application, model_path)


def score_application(application, model_path):
    if not isinstance(application, dict) or set(application) != set(FEATURES):
        raise ValueError("The JSON object must contain exactly the eleven fields in sample_application.json.")
    values = normalize_inputs(application)
    if any(value is None for value in values.values()):
        raise ValueError("Complete every application field.")
    if not 300 <= values["cibil_score"] <= 900:
        raise ValueError("Enter a CIBIL score between 300 and 900.")
    if any(values[key] <= 0 for key in ("income_annum", "loan_amount", "loan_term")):
        raise ValueError("Income, loan amount and loan term must be positive.")
    if values["no_of_dependents"] < 0 or not values["no_of_dependents"].is_integer():
        raise ValueError("Dependents must be a non-negative whole number.")
    for key, allowed in {"education": {"graduate", "not graduate"},
                         "self_employed": {"yes", "no"}}.items():
        if values[key] not in allowed:
            raise ValueError(f"Choose a valid value for {key.replace('_', ' ')}.")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    ranges = model.get("financial_ranges")
    if ranges is None:
        raise ValueError("Retrain the model with python loan_approval.py train to enable range checks.")
    issues = []
    for name, label in (("income_annum", "Annual income"), ("loan_amount", "Requested loan amount")):
        low, high = ranges[name]
        if not low <= values[name] <= high:
            issues.append(f"{label} is outside the training range ({low:,.0f} to {high:,.0f}).")
    ratio = values["loan_amount"] / values["income_annum"]
    if ratio > ranges["max_loan_to_income"]:
        issues.append(f"Loan-to-annual-income ratio is {ratio:,.2f}×. "
                      f"The highest training example was {ranges['max_loan_to_income']:.2f}×.")
    if issues:
        return {
            "status": "outside_training_range",
            "loan_to_income": round(ratio, 2),
            "reasons": issues,
        }
    encoder = model["encoder"]
    weights = model["weights"]
    encoded = encode(tuple(values.values()), encoder)
    score = predict(encoded, weights)

    # Combine the one-hot terms for each field on the model's log-odds scale.
    contributions = []
    offset = 1  # Skip the intercept, which applies to every application.
    for name in FEATURES:
        width = 1 if name in NUMERIC else len(encoder[name]["categories"]) + 1
        effect = sum(encoded[index] * weights[index] for index in range(offset, offset + width))
        contributions.append({"field": name, "effect": effect})
        offset += width
    contributions.sort(key=lambda factor: abs(factor["effect"]), reverse=True)
    return {
        "status": "estimated",
        "loan_to_income": round(ratio, 2),
        "demo_prediction": "Approved" if score >= 0.5 else "Rejected",
        "model_approval_score": score,
        "top_factors": contributions[:3],
        "threshold": 0.5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, default=DEFAULT_DATA)
    training.add_argument("--metrics", type=Path, default=METRICS_PATH)
    training.add_argument("--model", type=Path, default=MODEL_PATH)
    demo = commands.add_parser("predict")
    demo.add_argument("application", type=Path)
    demo.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    try:
        if args.command == "train":
            result = train(args.data.resolve(), args.model, args.metrics)
        else:
            result = predict_application(args.application, args.model)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
