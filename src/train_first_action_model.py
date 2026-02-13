import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA_PATH = "outputs/train_features.csv"

FEATURES = [
    "L_speed_mean", "L_speed_max",
    "R_speed_mean", "R_speed_max",
    "distance_closing"
]

def main():
    df = pd.read_csv(DATA_PATH)

    if "y_first_action_is_R" not in df.columns:
        raise ValueError("Missing y_first_action_is_R column in outputs/train_features.csv")

    # Ensure we have both classes somewhere in the dataset
    classes = sorted(df["y_first_action_is_R"].unique().tolist())
    print("Classes present:", classes)

    if len(classes) < 2:
        raise ValueError("Need at least 2 classes overall (0 and 1) to train.")

    X = df[FEATURES]
    y = df["y_first_action_is_R"]

    # With tiny datasets, do NOT split (split can drop a class from training)
    if len(df) < 6:
        print(f"Dataset is tiny (n={len(df)}). Training on ALL data (no test split).")
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        print("\n✅ Model trained.")
        print("Intercept:", model.intercept_)
        print("Coefficients:")
        for name, coef in zip(FEATURES, model.coef_[0]):
            print(f"  {name:16s} {coef:+.4f}")

        # Show training predictions (just sanity check)
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

        out = df[["clip", "y_first_action_is_R"]].copy()
        out["pred_prob_R"] = probs
        out["pred_y"] = preds
        print("\nTraining-set predictions (sanity check):")
        print(out.to_string(index=False))
        return

    # (We’ll re-enable a proper train/test split once you have more labels.)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n=== First Action AI Model Report ===")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()
