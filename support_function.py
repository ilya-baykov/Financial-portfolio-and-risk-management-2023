import numpy as np
import pandas as pd
from exception import BadValue


def neutralization(alpha: pd.DataFrame()) -> np.ndarray:
    return np.array(alpha) - np.array(alpha).mean()


def normalization(alpha: pd.DataFrame()) -> np.ndarray:
    return np.array(alpha) / np.abs(alpha).sum()


def neutralization_with_normalization(alpha: pd.DataFrame()) -> pd.DataFrame():
    for i in range(len(alpha.columns)):
        alpha[alpha.columns[i]] = normalization(neutralization(alpha[alpha.columns[i]]))
    return alpha


def calculation_profit(dataframe: pd.DataFrame()) -> pd.DataFrame():
    profit = pd.DataFrame()
    for i in range(len(dataframe.columns) - 2):
        profit[dataframe.columns[i + 2]] = dataframe[dataframe.columns[i + 2]] / dataframe[dataframe.columns[i + 1]] - 1
    return profit


def build_alpha(dataframe: pd.DataFrame()) -> pd.DataFrame():
    alpha = pd.DataFrame()
    for i in range(len(dataframe.columns) - 7):
        alpha[dataframe.columns[i + 7]] = -dataframe[dataframe.columns[i + 6]] / dataframe[dataframe.columns[i + 1]] + 1
    return alpha


def verification_alpha(alpha: pd.DataFrame()) -> None:
    for i in range(len(alpha.columns)):
        if alpha[alpha.columns[i]].sum() > 0.0000001:
            print('error')
            raise BadValue
        if alpha[alpha.columns[i]].abs().sum() > 1.0000001:
            raise BadValue


def holding_pnl(alpha: pd.DataFrame(), profit: pd.DataFrame()) -> pd.Series():
    profit_matrix = pd.DataFrame()
    for i in range(len(alpha.columns) - 1):
        profit_matrix[alpha.columns[i + 1]] = alpha[alpha.columns[i]] * profit[alpha.columns[i + 1]]
    return profit_matrix.sum(axis=0)


def turnover(alpha: pd.DataFrame()) -> pd.Series():
    turnover_matrix = pd.DataFrame()
    for i in range(len(alpha.columns) - 1):
        turnover_matrix[alpha.columns[i + 1]] = abs(alpha[alpha.columns[i + 1]] - alpha[alpha.columns[i]])
    return turnover_matrix.sum(axis=0)


def sharp(profit_vector: pd.DataFrame()) -> np.float64:
    return np.sqrt(len(profit_vector[-252:])) * np.mean(profit_vector) / np.std(profit_vector)


def cumm_profit(profit: pd.DataFrame()) -> pd.Series():
    return profit.cumsum()


def draw_down(cumm_profit_data: pd.Series()):
    max_draw_down = 0
    peak_value = cumm_profit_data[0]
    for current_value in cumm_profit_data:
        if current_value > peak_value:
            peak_value = current_value
        else:
            current_draw_down = peak_value - current_value
            max_draw_down = max(max_draw_down, current_draw_down)

    return max_draw_down
