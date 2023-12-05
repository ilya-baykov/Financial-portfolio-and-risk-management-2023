import random

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


def truncate(alpha: pd.DataFrame(), threshold: float) -> np.ndarray:
    _alpha = np.array(alpha)
    # Используется булев массив для индексации массива _alpha.
    # Это означает, что будут выбраны только те элементы _alpha,
    # для которых соответствующий элемент в булевом массиве равен True.
    # Таким образом, выбираются все элементы в _alpha, которые больше threshold.
    _alpha[_alpha > threshold] = threshold
    _alpha[_alpha < -threshold] = -threshold
    _alpha[_alpha >= 0] /= (2 * np.sum(_alpha[_alpha >= 0]))
    _alpha[_alpha < 0] /= (2 * np.sum(_alpha[_alpha < 0]))
    return _alpha


def rank(lst: list):
    lst_pos = list(enumerate(lst))
    lst_pos.sort(key=lambda x: x[1])

    for i in range(len(lst_pos)):
        lst_pos[i] = list(lst_pos[i])

    for i in range(len(lst_pos)):
        lst_pos[i].append((i + 1) / len(lst_pos))

    lst_pos.sort(key=lambda x: x[0])

    return [i[2] for i in lst_pos]


def cut_outliers(lst: list, n: int) -> list:
    def exceptional_values():
        sort_lst = sorted(lst)
        min_elements = sort_lst[:n:]
        sort_lst.reverse()
        max_elements = sort_lst[:n:]
        return min_elements + max_elements

    def element_exclusion():
        exceptional = exceptional_values()
        result = lst.copy()
        while exceptional:
            current_exception_element = exceptional[0]
            pos = result.index(current_exception_element)
            result[pos] = 0
            exceptional.remove(current_exception_element)
        return result

    return element_exclusion()


def cut_middle(lst: list, n: int) -> list:
    def exceptional_values():
        sort_lst = sorted(lst)
        left_side = sort_lst[:len(lst) // 2:]
        right_side = sort_lst[len(lst) // 2::]
        middle = left_side[-n // 2::] + right_side[:n // 2:]
        return middle

    def element_exclusion():
        exceptional = exceptional_values()
        result = lst.copy()
        while exceptional:
            current_exception_element = exceptional[0]
            pos = result.index(current_exception_element)
            result[pos] = 0
            exceptional.remove(current_exception_element)
        return result

    return element_exclusion()


def profit_correlation(alpha_1: pd.DataFrame(), alpha_2: pd.DataFrame()):
    profit_1 = holding_pnl(alpha_1, alpha_2)
    profit_2 = profit_1.copy()
    for i in range(len(profit_2)):
        profit_2[i] -= random.random()
    return np.corrcoef(profit_1, profit_2)[0][1]
