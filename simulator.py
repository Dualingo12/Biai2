import numpy as np


def simulate_trading(preds, prices, threshold=0.3, tp=0.01, sl=0.005):
    balance = 1000.0
    position = 0
    entry_price = 0.0
    trades = []
    equity_curve = [1000.0]

    for prob, price in zip(preds, prices):
        action = int(np.argmax(prob))

        if position == 0:
            if action == 1 and prob[1] > threshold:
                position = 1
                entry_price = float(price)
            elif action == 2 and prob[2] > threshold:
                position = -1
                entry_price = float(price)

        elif position == 1:
            change = (float(price) - entry_price) / entry_price
            if change >= tp or change <= -sl:
                balance *= 1 + change
                trades.append(change)
                position = 0

        elif position == -1:
            change = (entry_price - float(price)) / entry_price
            if change >= tp or change <= -sl:
                balance *= 1 + change
                trades.append(change)
                position = 0

        equity_curve.append(balance)

    if position != 0:
        final_price = float(prices[-1])
        if position == 1:
            change = (final_price - entry_price) / entry_price
        else:
            change = (entry_price - final_price) / entry_price
        balance *= 1 + change
        trades.append(change)

    equity_curve.append(balance)

    return balance, trades, equity_curve
