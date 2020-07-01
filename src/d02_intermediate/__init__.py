from sklearn.pipeline import Pipeline

import src.d01_data as d01
from src.d02_intermediate.AgePurger import AgePurger
from src.d02_intermediate.CabinPurger import CabinPurger
from src.d02_intermediate.EmbarkedPurger import EmbarkedPurger

df = d01.load_data('01_raw', 'train')

print('Dimension of dataset before purging "{}": '.format('train'), df.shape)
print(df.info())

pipeline = Pipeline([
        ('age_purger', AgePurger()),
        ('cabin_purger', CabinPurger()),
        ('embarked_purger', EmbarkedPurger())
    ])

df = pipeline.fit_transform(df)
d01.write_data(df, '02_intermediate', 'train')

print('Dimension of dataset after purging "{}": '.format('train'), df.shape)
print(df.info())
