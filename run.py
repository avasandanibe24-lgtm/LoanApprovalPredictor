"""Train the default model if needed, then launch the dashboard."""
import dashboard
from loan_approval import DEFAULT_DATA, METRICS_PATH, MODEL_PATH, train


def main():
    # Train only when the saved model or its evaluation is missing.
    if not (MODEL_PATH.is_file() and METRICS_PATH.is_file()):
        print("Training the model with the Kaggle Loan Approval dataset...", flush=True)
        try:
            train(DEFAULT_DATA, MODEL_PATH, METRICS_PATH)
        except (OSError, ValueError) as error:
            raise SystemExit(
                f"Training failed: {error}\n"
                "If the download failed, save loan_approval_dataset.csv beside run.py and try again. "
                "The README contains the download link."
            ) from error
    dashboard.main()


if __name__ == "__main__":
    main()
