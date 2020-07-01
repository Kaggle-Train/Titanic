from src.d03_processing.title_processor import TitleProcessor
from src.d03_processing.mother_child_processor import MotherChildProcessor
from sklearn.pipeline import Pipeline
import src.d01_data as d01

df = d01.load_data('02_intermediate', 'train')

pipeline = Pipeline([
    ('title_processor', TitleProcessor()),
    ('mother_child_processor', MotherChildProcessor())
])

df = pipeline.fit_transform(df)
d01.write_data(df, '03_processed', 'train')

print('Dimension of dataset after processing "{}": '.format('train'), df.shape)
print(df.info())