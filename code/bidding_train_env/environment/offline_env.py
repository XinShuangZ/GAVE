import numpy as np

class OfflineEnv:
    def __init__(self, min_remaining_budget: float = 0.1):
        self.min_remaining_budget = min_remaining_budget

    def simulate_ad_bidding(
        self,
        pValues: np.ndarray, 
        pValueSigmas: np.ndarray,
        bids: np.ndarray, 
        leastWinningCosts: np.ndarray
    ):
        tick_status = bids >= leastWinningCosts
        tick_cost = leastWinningCosts * tick_status
        values = np.random.normal(loc=pValues, scale=pValueSigmas)
        values = values*tick_status
        tick_value = np.clip(values,0,1)
        tick_conversion = np.random.binomial(n=1, p=tick_value)
        return tick_value, tick_cost, tick_status, tick_conversion

def test():
    pv_values = np.array([0.8, 0.2, 0.6, 0.5, 0.7])
    pv_values_sigma = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    bids = np.array([15, 20, 35, 45, 55])
    market_prices = np.array([12, 22, 32, 42, 52]) # leastWinningCosts

    env = OfflineEnv()
    tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
        pv_values,
        pv_values_sigma,
        bids,
        market_prices
    )
    print(f"Tick Value: {tick_value}")
    print(f"Tick Cost: {tick_cost}")
    print(f"Tick Status: {tick_status}")
    print(f"Tick Conversions: {tick_conversion}")


if __name__ == '__main__':
    test()

