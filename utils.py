from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt

def split_sets(dataset):
    X = dataset.drop(columns='DEATH_EVENT')
    y = dataset['DEATH_EVENT']

    X_train, X_test, y_train, y_test= train_test_split(X, y, stratify=y, random_state=20, test_size= 0.2)

    scaler= MinMaxScaler()

    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def plot_precision_recall_curve(clf, X_test, y_test):
    predictions = clf.predict(X_test)
    precision, recall, _ = precision_recall_curve(y_test, predictions)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.show()

    return disp
