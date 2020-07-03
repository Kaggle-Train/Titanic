from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import src.d01_data as d01
import joblib
from src.d05_evaluation.evaluation_summary import EvaluationSummary

DATE = '20200701'

df = d01.load_data('03_processed', 'train')

### INTERACTIONS ###
df['Sex_female TicketProbability'] = df.Sex_female * df.TicketProbability
df['Pclass Sex_female'] = df.Pclass * df.Sex_female

X = df[['Age', 'Fare', 'FamilySize', 'Pclass', 'Sex_female', 'MotherChildRelation', 'Embarked_C', 'Embarked_S', 
                  'TicketProbability', 'Cabin', 'IsRev', 'Sex_female TicketProbability', 'Pclass Sex_female']]
y = df.Survived

lr = LogisticRegression()
param_grid = [
    {'penalty': ['none', 'l2'], 'max_iter': [1000, 5000, 9999]}
    ]
grid_search = GridSearchCV(lr, param_grid, scoring='neg_mean_squared_error', return_train_score=True, cv=5)
grid_search.fit(X,y)
d01.write_model(grid_search, '20200703_LR')
lr = grid_search.best_estimator_


knn = KNeighborsClassifier()
param_grid = [
    {'n_neighbors': range(15,20,1)}
    ]
grid_search = GridSearchCV(knn, param_grid, scoring='neg_mean_squared_error', return_train_score=True, cv=5)
grid_search.fit(X,y)
d01.write_model(grid_search, '20200703_KNN')
knn = grid_search.best_estimator_

# Support Vector Machines
svc = SVC(probability=True)
# Decision Tree
dt = DecisionTreeClassifier(max_depth=3)
# Random Forest
rf = RandomForestClassifier(max_depth=3)

mlp = MLPClassifier()
param_grid = [
    {'hidden_layer_sizes': [(10)], 'max_iter': [1000]}
    ]
grid_search = GridSearchCV(mlp, param_grid, scoring='neg_mean_squared_error', return_train_score=True, cv=5)
grid_search.fit(X,y)
d01.write_model(grid_search, '20200703_MLP')
mlp = grid_search.best_estimator_

vo = VotingClassifier([('lr', lr), ('knn', knn), ('svc', svc), ('dt', dt), ('rf', rf), ('mlp', mlp)])
param_grid = [
    {'voting': ['soft']}
    ]
grid_search = GridSearchCV(vo, param_grid, scoring='neg_mean_squared_error', return_train_score=True, cv=5)
grid_search.fit(X,y)
d01.write_model(grid_search, '20200703_VO')
vo = grid_search.best_estimator_

models = [lr, knn,svc, dt, rf, mlp, vo]

for model in models:
    summary = EvaluationSummary(model, X, y)
    print("Train: %0.5f,  Cross accuracy validation: %0.3f (+/- %0.3f)  {}".format(type(model).__name__) % (
        summary.train_accuracy, summary.scores.mean(), summary.scores.std() * 2))