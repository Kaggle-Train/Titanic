from src.d03_processing.general.mean_encoder import MeanEncoder
from src.d03_processing.title_processor import TitleProcessor
from src.d03_processing.mother_child_processor import MotherChildProcessor
from src.d03_processing.family_processor import FamilyProcessor
from src.d03_processing.sex_processor import SexProcessor
from src.d03_processing.ticket_processor import TicketProcessor
from src.d03_processing.embarked_processor import EmbarkedProcessor
from sklearn.pipeline import Pipeline
import src.d01_data as d01
from sklearn import preprocessing


def run(data_type):
    if (data_type == 'train' or data_type == 'eval'):
        df = d01.load_data('02_intermediate', 'train')
        pipeline = Pipeline([
            ('title_processor', TitleProcessor()),
            ('mother_child_processor', MotherChildProcessor(data_type=data_type)),
            ('family_processor', FamilyProcessor()),
            ('sex_processor', SexProcessor()),
            ('ticket_processor', TicketProcessor(data_type=data_type)),
            ('family_survived_processor',
             MeanEncoder(add_feature=True, new_feature_name='FamilyProbability', mean_feature_name='LastName',
                         target_name='Survived', data_type='train')),
            ('embarked_processor', EmbarkedProcessor())
        ])
    if data_type == 'test':
        df = d01.load_data('02_intermediate', 'test')
        df_eval = d01.load_data('03_processed', 'eval')
        pipeline = Pipeline([
            ('title_processor', TitleProcessor()),
            ('mother_child_processor', MotherChildProcessor(data_type=data_type, data_eval=df_eval)),
            ('family_processor', FamilyProcessor()),
            ('sex_processor', SexProcessor()),
            ('ticket_processor', TicketProcessor(data_type=data_type, data_eval=df_eval)),
            ('family_survived_processor',
             MeanEncoder(add_feature=True, new_feature_name='FamilyProbability', mean_feature_name='LastName',
                         target_name='Survived', data_type='train', data_eval=df_eval)),
            ('embarked_processor', EmbarkedProcessor())
        ])

    df = pipeline.fit_transform(df)

    # Perform scaling after processing since numerical features are also used for transforming data
    df.loc[:, 'Age'] = preprocessing.scale(df.Age)
    df.loc[:, 'Fare'] = preprocessing.scale(df.Fare)
    df.loc[:, 'FamilySize'] = preprocessing.scale(df.FamilySize)
    df.loc[:, 'Pclass'] = preprocessing.scale(df.Pclass)

    d01.write_data(df, '03_processed', data_type)

    print('Dimension of dataset after processing "{}": '.format(data_type), df.shape)
    print(df.info())

    return df


run('train')
