from support_function import *
from drawing_graphs import *
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('Close.csv')

profit = calculation_profit(df)

alpha = build_alpha(df)

alpha = neutralization_with_normalization(alpha)

verification_alpha(alpha)

alpha_yield_vector = holding_pnl(alpha, profit)

turnover_alpha = turnover(alpha)

sharp_alpha = sharp(holding_pnl(alpha[-252:], profit))

cumm_profit_value = cumm_profit(alpha_yield_vector)

# cumm_profit_graphs(cumm_profit_value)

draw_down_value = draw_down(cumm_profit_value)

# alpha_stats_graphs(alpha, profit)
# instrument_volatility(profit)
print(profit_correlation(alpha, profit))
