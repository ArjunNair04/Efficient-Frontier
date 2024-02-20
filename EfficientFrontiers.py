import pandas as pd
import numpy as np
import scipy.optimize as sco

def get_portfolio_return(weights, returns):
    """ Calculate the expected portfolio return.
    
    :param weights: numpy.ndarray
        The asset weights in the portfolio.
    :param returns: numpy.ndarray
        The expected returns of the assets.
    :return: float
        The expected return of the portfolio.
    """
    return np.dot(weights, returns)

def get_portfolio_volatility(weights, cov_matrix):
    """ Calculate the expected portfolio volatility.
    
    :param weights: numpy.ndarray
        The asset weights in the portfolio.
    :param cov_matrix: numpy.ndarray
        The covariance matrix of the asset returns.
    :return: float
        The expected volatility of the portfolio.
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def get_efficient_frontier(returns, cov_matrix, return_targets):
    """ Calculate the efficient frontier using optimization.
    
    :param returns: numpy.ndarray
        The expected returns of the assets.
    :param cov_matrix: numpy.ndarray
        The covariance matrix of the asset returns.
    :param return_targets: numpy.ndarray
        The range of target returns to optimize for.
    :return: pd.DataFrame
        The efficient frontier as a DataFrame.
    """
    efficient_portfolios = []
    num_assets = len(returns)
    args = (cov_matrix,)
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.ones(num_assets) / num_assets

    for target_return in return_targets:
        constraints = (
            {'type': 'eq', 'fun': lambda x: get_portfolio_return(x, returns) - target_return},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        )
        result = sco.minimize(get_portfolio_volatility, initial_guess, args=args, method='SLSQP',
                              constraints=constraints, bounds=bounds)
        efficient_portfolios.append(result)

    volatilities = [portfolio['fun'] for portfolio in efficient_portfolios]
    efficient_frontier_df = pd.DataFrame({'returns': return_targets, 'volatility': volatilities})
    return efficient_frontier_df

def simulate_efficient_frontier(returns, cov_matrix, num_portfolios=100000):
    """ Simulate the efficient frontier using random portfolio weights.
    
    :param returns: numpy.ndarray
        The expected returns of the assets.
    :param cov_matrix: numpy.ndarray
        The covariance matrix of the asset returns.
    :param num_portfolios: int
        The number of portfolios to simulate.
    :return: dict
        A dictionary containing the simulated portfolios and the efficient frontier.
    """
    num_assets = len(returns)
    weights = np.random.random(size=(num_portfolios, num_assets))
    weights /= np.sum(weights, axis=1)[:, np.newaxis]
    
    portfolio_returns = np.dot(weights, returns)
    portfolio_volatilities = np.array([get_portfolio_volatility(weight, cov_matrix) for weight in weights])

    simulated_df = pd.DataFrame({'returns': portfolio_returns, 'volatility': portfolio_volatilities})
    
    return_points = 300
    frontier_volatilities = []
    return_grid = np.linspace(simulated_df.returns.min(), simulated_df.returns.max(), return_points)
    
    for return_level in return_grid:
        if return_level in portfolio_returns:
            matched_indices = np.where(portfolio_returns == return_level)
            frontier_volatilities.append(np.min(portfolio_volatilities[matched_indices]))

    efficient_frontier_df = pd.DataFrame({'returns': return_grid, 'volatility': frontier_volatilities})
    return {'Simulated_Portfolios': simulated_df, 'Efficient_Frontier': efficient_frontier_df}