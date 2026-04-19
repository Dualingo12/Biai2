import pandas as pd


def load_csv(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna().reset_index(drop=True)

    if df.shape[1] >= 6 and not {"close", "volume"}.issubset(df.columns):
        df = df.iloc[:, :6]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        except Exception:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df
