import matplotlib.pyplot as plt

from support_function import *


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


def alpha_stats_graphs(_alpha: pd.DataFrame(), _profit) -> None:
    def plot_profit():
        """Строит график доходности."""
        plt.figure(figsize=(25, 10))
        plt.title('Profit')
        plt.xlabel('Date')
        plt.ylabel('Profit')
        plt.grid()
        _alpha_yield_vector = holding_pnl(_alpha, _profit)
        plt.plot(_alpha_yield_vector)
        plt.legend(['Profit'], loc='upper left')
        plt.xticks(np.arange(-10, len(_alpha_yield_vector) + 10, 100))
        plt.show()

    def get_unique_years():
        """Возвращает уникальные года в данных."""
        return sorted(set(col[:4] for col in _alpha.columns[1:]))

    def get_columns_by_year(year):
        """Возвращает столбцы матрицы альфа для заданного года."""
        return [col for col in _alpha.columns[1:] if col.startswith(year)]

    def calculate_yearly_stats(years):
        """Вычисляет статистику для каждого года."""
        stats = {'sharpes': [], 'turnovers': [], 'cumprofits': [], 'drawdowns': [], 'years': []}

        for year in years:
            year_columns = get_columns_by_year(year)
            year_alpha = _alpha[year_columns]

            stats['sharpes'].append(sharp(holding_pnl(year_alpha, _profit)))
            stats['turnovers'].append(turnover(year_alpha).values.mean())
            stats['cumprofits'].append(cumm_profit(holding_pnl(year_alpha, _profit))[-1])
            stats['drawdowns'].append(draw_down(cumm_profit(holding_pnl(year_alpha, _profit))))
            stats['years'].append(year)

        return pd.DataFrame(stats).set_index('years')

    def alpha_stats():
        plot_profit()

        years = get_unique_years()
        stats = calculate_yearly_stats(years)
        print(stats)
        return stats

    alpha_stats()
