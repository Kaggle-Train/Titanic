from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import src.d01_data as d01
import joblib

DATE = '20200701'

df = d01.load_data('03_processed', 'train')

### INTERACTIONS ###
df['Sex_female TicketProbability'] = df.Sex_female * df.TicketProbability
df['Pclass Sex_female'] = df.Pclass * df.Sex_female

X = df[['Age', 'Fare', 'FamilySize', 'Pclass', 'Sex_female', 'MotherChildRelation', 'Embarked_C', 'Embarked_S', 
                  'TicketProbability', 'Cabin', 'IsRev', 'Sex_female TicketProbability', 'Pclass Sex_female']]
y = df.Survived

param_grid = [
    {'penalty': ['none', 'l2'], 'max_iter': [1000, 5000, 9999]}
    ]



lr = LogisticRegression()
grid_search = GridSearchCV(lr, param_grid, scoring='neg_mean_squared_error', return_train_score=True, cv=5)
grid_search.fit(X,y)
print(grid_search.best_estimator_.score(X,y))

lr2 = LogisticRegression()
lr2.fit(X, y)
print(lr2.score(X, y))

print(grid_search.best_estimator_)
print(lr2)