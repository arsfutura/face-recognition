import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from dataset import load_data
from constants import MODEL_PATH
from sklearn.ensemble import RandomForestClassifier

CLASSIFIERS = {
    'svm': (
        SVC(decision_function_shape='ovr'),
        [
            {'C': [0.001, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [0.001, 0.1, 1, 10, 100, 1000], 'gamma': [100, 50, 10, 1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
        ]
    ),
    'logreg': (
        LogisticRegression(max_iter=100000),
        {'C': [0.001, 0.1, 1, 10, 100, 1000]}
    )
}


def main():
    X, y = load_data()
    le = LabelEncoder()
    le.fit(y)

    classifier = CLASSIFIERS['logreg']

    clf = GridSearchCV(classifier[0], classifier[1], cv=5, verbose=1)
    clf.fit(X, le.transform(y))

    print clf.best_params_
    print classification_report(le.transform(y), clf.predict(X), target_names=le.classes_)
    pickle.dump((le, clf), open(MODEL_PATH, 'wb'))


if __name__ == '__main__':
    main()
