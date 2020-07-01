import joblib

def write_model(model, name):
    joblib.dump(model, '../../data/04_models/{}.pkl'.format(name))