import src.d01_data.load_data as d01_load_data
import src.d01_data.write_data as d01_write_data


def load_data(level, name):
    return d01_load_data.load_data(level, name)


def write_data(df, level, name):
    return d01_write_data.write_data(df, level, name)
