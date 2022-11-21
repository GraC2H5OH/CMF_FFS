def preprocessing(data):
    data['Equity Charge'] = [i for i in range(len(data['Equity Charge']))]
    y = data['Revenue']
    data.drop(columns=['Revenue'], inplace=True)
    
    return data, y
