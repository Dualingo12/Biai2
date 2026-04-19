import glob
import os
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F

from load_data import load_csv
from prepare_data import prepare_dataset
from model import TradingModel
from metrics import compute_metrics
from simulator import simulate_trading


def save_models(best_models, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    for i, (balance, state) in enumerate(best_models, start=1):
        file_name = f"model_top_{i}_{int(balance)}.pt"
        path = os.path.join(out_dir, file_name)
        torch.save(state, path)


def print_class_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique.tolist(), counts.tolist()))
    print("Class distribution:", distribution)


def train(model, X_train, y_train, prices_train, X_val, y_val, prices_val, epochs=1, lr=1e-3, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_balance = 0.0
    best_models = []

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)

    batch_size = 64

    for epoch in range(epochs):
        model.train()

        # Batch training
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i:i+batch_size]
            y_batch = y_train_tensor[i:i+batch_size]

            outputs = model(X_batch)
            loss = F.cross_entropy(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(X_val_tensor), dim=1).cpu().numpy()

        balance, trades, equity_curve = simulate_trading(probs, prices_val)
        reward = (balance - 1000) / 1000  # normalized profit

        loss = loss * (1 - reward)  # усиливаем или ослабляем loss через reward

        metrics = compute_metrics(trades, equity_curve)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"loss={loss.item():.4f}, "
            f"balance={balance:.2f}, "
            f"trades={metrics['trades']}, "
            f"winrate={metrics['winrate']:.2f}, "
            f"avg_return={metrics['avg_return']:.4f}, "
            f"profit_factor={metrics['profit_factor']:.4f}, "
            f"max_drawdown={metrics['max_drawdown']:.4f}"
        )

        if balance > best_balance:
            best_balance = balance
            best_models.append((balance, copy.deepcopy(model.state_dict())))
            best_models = sorted(best_models, key=lambda item: item[0], reverse=True)[:3]

    return best_models


def main():
    files = sorted(glob.glob("data/*.csv"))
    if not files:
        raise RuntimeError("No CSV files found in the data directory.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradingModel().to(device)

    best_models = []
    max_train_time = 3600
    start_time = time.time()

    file_index = 0
    while time.time() - start_time < max_train_time:
        file_path = files[file_index]
        file_index = (file_index + 1) % len(files)

        df = load_csv(file_path)
        X, y, prices = prepare_dataset(df, seq_len=10, threshold=0.002)

        if len(y) == 0:
            continue

        # Sync prices with X
        prices = prices[len(prices) - len(X):]  # align prices to X length

        # Train/val split
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        prices_train, prices_val = prices[:split], prices[split:]

        print(f"Training on {os.path.basename(file_path)} - samples={len(y_train)}")
        print_class_distribution(y_train)

        if np.unique(y_train).size < 2:
            print("Not enough class variety in this dataset, skipping file.")
            continue

        file_best_models = train(model, X_train, y_train, prices_train, X_val, y_val, prices_val, epochs=1, lr=1e-3, device=device)

        for balance, state in file_best_models:
            best_models.append((balance, state))
        best_models = sorted(best_models, key=lambda item: item[0], reverse=True)[:3]

        save_models(best_models, out_dir=".")

    print("Training complete.")
    print("Top models saved:")
    for rank, (balance, _) in enumerate(best_models, start=1):
        print(f"{rank}: balance={balance:.2f}")


if __name__ == "__main__":
    main()
