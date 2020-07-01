def write_data(df, level, name):
    df.to_csv('../../data/{}/{}.csv'.format(level, name))
