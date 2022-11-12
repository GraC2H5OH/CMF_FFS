from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score


def wape(y, pred):
    return (y - pred).abs().sum() / y.sum()


def r2_adj(y, pred, p):
    r2 = r2_score(y, pred)
    n = len(y)
    return 1 - ((1 - r2) * (n - 1)) / (n - p - 1)


def metrics(y, pred, p, one_model=True):
    if one_model:
        print('MAPE =', mean_absolute_percentage_error(y, pred))
        print('WAPE =', wape(y, pred))
        print('MSE =', mean_squared_error(y, pred))
        print('R2_adj', r2_adj(y, pred, p))
	return
    else:
        mape, wape_multi, mse, r2_adj_multi = [0] * 4
        n = len(y)
        for i, j, k in zip(y, pred, p):
            mape += mean_absolute_percentage_error(i, j)
            wape_multi += wape(i, j)
            mse += mean_squared_error(i, j)
            r2_adj_multi += r2_adj(i, j, k)
    	print('Mean MAPE =', mape / n)
    	print('Mean WAPE =', wape_multi / n)
    	print('Mean MSE =', mse / n)
    	print('Mean R2_adj', r2_adj_multi / n)