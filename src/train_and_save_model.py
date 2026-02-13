import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_PATH = "outputs/train_features.csv"
MODEL_PATH = "outputs/first_action_model.joblib"

FEATURES = [
    "L_speed_mean", "L_speed_max",
    "R_speed_mean", "R_speed_max",
    "distance_closing"
]

def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df["y_first_action_is_R"]

    if y.nunique() < 2:
        print("Need both classes (0 and 1) to train.")
        return

    if len(df) < 10:
        print(f"n={len(df)} is small. Training on ALL data, saving model.")
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        joblib.dump(model, MODEL_PATH)
        print(f"Saved model -> {MODEL_PATH}")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")

if __name__ == "__main__":
    main()
