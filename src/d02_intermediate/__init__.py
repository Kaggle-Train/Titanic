import src.d01_data as d01
from src.d02_intermediate.AgePurger import AgePurger

df = d01.load_data('01_raw', 'train')

age_purger = AgePurger()
df = age_purger.transform(df)

d01.write_data(df, '02_intermediate', 'train')

print('Dimension of dataset after purging"{}": '.format('train'), df.shape)
