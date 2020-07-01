import src.d01_data.load_data as d01_load_data
import src.d01_data.write_data as d01_write_data
import src.d01_data.write_model as d01_write_model
import src.d01_data.load_model as d01_load_model


def load_data(level, name):
    return d01_load_data.load_data(level, name)

def write_data(df, level, name):
    d01_write_data.write_data(df, level, name)

def load_model(name):
    return d01_load_model.load_model(name)

def write_model(model, name):
    d01_write_model.write_model(model, name)
    