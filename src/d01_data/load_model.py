import joblib

def load_model(name):
    return joblib.load('../../data/04_models/{}.pkl'.format(name))