import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import  roc_auc_score


def fit_model(data, model_clf, test_size, random_state):
    x = data.x
    y = data.y
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, 
                                                        random_state=random_state)
    model_clf.fit(x_train, y_train) 

    model_predict_data_train = model_clf.predict(x_train)
    model_predict_data_test = model_clf.predict(x_test)

    model_accuracy_train = np.mean(model_predict_data_train==y_train)
    model_accuracy_test = np.mean(model_predict_data_test==y_test)

    print("Accuracy of training set:")
    print ( model_accuracy_train)
    print("Accuracy of test set:")
    print (model_accuracy_test)

    # summarize the fit of the train model
    model_expected_train = y_train
    model_predicted_train = model_clf.predict(x_train)
    print("Training set: confusion matrix:")
    print(metrics.classification_report(model_expected_train, model_predicted_train))
    print(metrics.confusion_matrix(model_expected_train, model_predicted_train))

    # summarize the fit of the test model
    model_expected = y_test
    model_predicted = model_clf.predict(x_test)
    print("Test set: confusion matrix:")
    print(metrics.classification_report(model_expected, model_predicted))
    print(metrics.confusion_matrix(model_expected, model_predicted))


def save_model(model_clf, path):
    joblib.dump(model_clf, path)


def model_predict(data, path):
    x_validation = data.x
    y_validation = data.y

    model_clf = joblib.load(filename=path)
    model_predict_data_validation = model_clf.predict(x_validation)
    model_accuracy_validation = np.mean(model_predict_data_validation==y_validation)
    
    print("Accuracy of validation set:")
    print (model_accuracy_validation)

    # summarize the fit of the test model
    model_expected = y_validation
    model_predicted = model_clf.predict(x_validation)
    print("Validation set: confusion matrix:")
    print(metrics.classification_report(model_expected, model_predicted))
    print(metrics.confusion_matrix(model_expected, model_predicted))
    print("MCC", matthews_corrcoef(model_expected, model_predicted))
    print( "AUC:", roc_auc_score(model_expected, model_predicted))




