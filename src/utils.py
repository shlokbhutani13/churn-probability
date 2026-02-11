import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    return df


def normalize_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures there's a 'target' column with 0/1 values.
    Accepts Yes/No, True/False, 1/0, etc.
    """
    target_col = None
    for cand in ["Churn", "churn", "Exited", "exit", "Attrition", "attrition"]:
        if cand in df.columns:
            target_col = cand
            break

    if target_col is None:
        raise ValueError(
            "Could not find a target column. Expected something like 'Churn' or 'churn'. "
            f"Columns found: {df.columns.tolist()}"
        )

    y = df[target_col]

    # ALWAYS treat as string first (handles Yes/No safely)
    y_str = y.astype(str).str.strip().str.lower()

    mapping = {
        "yes": 1, "y": 1, "true": 1, "1": 1,
        "no": 0, "n": 0, "false": 0, "0": 0
    }
    df["target"] = y_str.map(mapping)

    if df["target"].isna().any():
        bad = df.loc[df["target"].isna(), target_col].unique()
        raise ValueError(f"Unrecognized target values in '{target_col}': {bad}")

    return df.drop(columns=[target_col])
