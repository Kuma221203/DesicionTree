
from sklearn import tree, preprocessing, metrics
import pandas as pd
from sklearn.model_selection import train_test_split

file = pd.read_csv('50_Startups.csv')
my_data = pd.DataFrame(file).to_numpy()
my_data.shape

label_encoder = preprocessing.LabelEncoder()
my_data[:,3] = label_encoder.fit_transform(my_data[:,3])
my_data

X = my_data[:,:4]
y = my_data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

my_prof = tree.DecisionTreeRegressor(min_samples_leaf = 4, min_samples_split = 4, random_state=0)
my_prof.fit(X_train, y_train)

y_predict = my_prof.predict(X_test)

print('Train score =', my_prof.score(X_train, y_train))
print('Test score =', my_prof.score(X_test, y_test))
print('The MAE is:', metrics.mean_absolute_error(y_predict, y_test))