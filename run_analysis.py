"""End-to-end runner: load -> preprocess -> tune -> evaluate -> write results."""

from src import config
from src.evaluate import evaluate_all
from src.load_data import class_balance, load_relabeled, missing_report, run_summary
from src.preprocess import prepare
from src.train import tune_models


def main():
    print("Loading relabeled data from", config.DATA_DIR)
    df = load_relabeled()
    print(f"  combined shape: {df.shape}")
    print(f"  runs (groups): {df[config.GROUP_COLUMN].nunique()}")
    print("\nClass balance per food type:")
    print(class_balance(df).to_string())
    print("\nRuns per food type:")
    print(run_summary(df).to_string(index=False))
    print("\nMissing values (numeric features):")
    print(missing_report(df).to_string())

    print("\nPreparing features (warm-up trim, label encode)...")
    X, y, groups = prepare(df)
    print(f"  X shape after trim: {X.shape}  | positive rate (spoiled): {y.mean():.3f}")

    print("\nTuning 4 models with GridSearchCV + StratifiedGroupKFold...")
    searches = tune_models(X, y, groups)

    print("\nEvaluating (out-of-fold)...")
    report, metrics_table, cv_summaries, best_name = evaluate_all(searches, X, y, groups)
    print("\n" + report)
    print(f"\nResults written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
