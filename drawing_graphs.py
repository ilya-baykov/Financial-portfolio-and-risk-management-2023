import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def cumm_profit_graphs(cumm_profit_data: pd.Series()) -> None:
    plt.figure(figsize=(25, 10))
    plt.title('Cumprofit')
    plt.xlabel('Date')
    plt.ylabel('Accumulated profit')
    plt.grid()
    plt.plot(cumm_profit_data)
    plt.legend(['Accumulated profit'], loc='upper left')
    plt.xticks(
        np.arange(
            -10,
            len(cumm_profit_data) + 10,
            100
        )
    )

    plt.show()
