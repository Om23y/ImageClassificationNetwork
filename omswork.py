from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from omsmodel import Net2



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



clf = RandomForestClassifier(n_estimators=100, random_state=0)

clf.fit(X_train,y_train)

training_accuracy = clf.score(X_train, y_train)
print(f'Training accuracy: {training_accuracy*100:.2f}%')


testing_accuracy = clf.score(X_test, y_test)
print(f'Testing accuracy: {testing_accuracy*100:.2f}%')
