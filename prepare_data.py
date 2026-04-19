import numpy as np

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    class StandardScaler:
        def fit(self, X):
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            self.scale_[self.scale_ == 0] = 1.0
            return self

        def transform(self, X):
            return (X - self.mean_) / self.scale_

        def fit_transform(self, X):
            return self.fit(X).transform(X)


def create_features(df):
    df = df.copy()

    if "close" not in df.columns or "volume" not in df.columns:
        raise ValueError("Input data must include close and volume columns")

    df["return"] = df["close"].pct_change()
    df["ma"] = df["close"].rolling(10).mean()
    df["std"] = df["close"].rolling(10).std()

    df = df.dropna().reset_index(drop=True)
    return df


def create_labels(df, threshold=0.002):
    future_return = df["close"].shift(-1) / df["close"] - 1
    labels = np.zeros(len(df), dtype=np.int64)
    labels[future_return > threshold] = 1
    labels[future_return < -threshold] = 2
    return labels[:-1]


def prepare_dataset(df, seq_len=10, threshold=0.002):
    df = create_features(df)

    prices = df["close"].values
    features = df[["close", "volume", "return", "ma", "std"]].values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X = []
    y = []
    sample_prices = []

    labels = create_labels(df, threshold=threshold)

    for i in range(len(features) - seq_len - 1):
        X.append(features[i : i + seq_len])
        y.append(labels[i + seq_len])
        sample_prices.append(prices[i + seq_len])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(sample_prices, dtype=np.float32)
