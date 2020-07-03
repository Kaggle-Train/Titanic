from sklearn.pipeline import Pipeline

import src.d01_data as d01
from src.d02_intermediate.age_purger import AgePurger
from src.d02_intermediate.cabin_purger import CabinPurger
from src.d02_intermediate.embarked_purger import EmbarkedPurger

def run(data_type):
    df = d01.load_data('01_raw', data_type)
    
    print('Dimension of dataset before purging "{}": '.format(data_type), df.shape)
    print(df.info())
    
    pipeline = Pipeline([
            ('age_purger', AgePurger()),
            ('cabin_purger', CabinPurger()),
            ('embarked_purger', EmbarkedPurger())
        ])
    
    df = pipeline.fit_transform(df)
    
    
    print('Dimension of dataset after purging "{}": '.format(data_type), df.shape)
    print(df.info())
    
    if(df.isna().sum().sum() > 0):
        print('ERROR: STILL NA VALUES IN DATA SET {}'.format(data_type))
        df['Age'] = df.Age.fillna(df.Age.mean())    
        df['Fare'] = df.Fare.fillna(df.Fare.mean()) 
        
        d01.write_data(df, '02_intermediate', data_type)
    else:
         d01.write_data(df, '02_intermediate', data_type)
         
    return df