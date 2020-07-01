from sklearn.pipeline import Pipeline

import src.d01_data as d01
from src.d02_intermediate.age_purger import AgePurger
from src.d02_intermediate.cabin_purger import CabinPurger
from src.d02_intermediate.embarked_purger import EmbarkedPurger

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
