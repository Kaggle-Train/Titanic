import pandas as pd
import os


def load_data(level, name):
    # DATA_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.abspath('')
    FILENAME = os.path.join(DATA_DIR, '../../data/{}/{}.csv'.format(level, name))

    print('Loading dataset {}.csv from {}'.format(name, level))

    df = pd.read_csv(FILENAME, index_col=0, low_memory=False)

    print('Dimension of dataset "{}": '.format(name), df.shape)

    return df
