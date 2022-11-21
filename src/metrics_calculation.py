from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score


def wape(y, pred):
    return (y - pred).abs().sum() / y.sum()


def metrics(y, pred, one_model=True):
    if one_model:
        mape = mean_absolute_percentage_error(y, pred)
        wape_t = wape(y, pred)
        mse = mean_squared_error(y, pred)
        return mape, wape_t, mse
    else:
        mape, wape_t, mse = 0, 0, 0
        n = len(y)

        for i, j in zip(y, pred):
            mape += mean_absolute_percentage_error(i, j)
            wape_t += wape(i, j)
            mse += mean_squared_error(i, j)
        return mape / n, wape_t / n, mse / n
