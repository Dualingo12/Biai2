import matplotlib.pyplot as plt
import numpy as np

def plot_signals(prices, signals, title="Trading Signals"):
    plt.figure(figsize=(14, 7))
    plt.plot(prices, label='Price', color='blue')
    
    buy_signals = signals == 1
    sell_signals = signals == 2
    
    plt.scatter(np.where(buy_signals)[0], prices[buy_signals], label='Buy Signal', marker='^', color='green', s=100)
    plt.scatter(np.where(sell_signals)[0], prices[sell_signals], label='Sell Signal', marker='v', color='red', s=100)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()