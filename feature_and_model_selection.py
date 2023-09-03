import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from utils import plot_precision_recall_curve
from sklearn.metrics import confusion_matrix
# plt.switch_backend('agg')
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def feature_selection(X_train, y_train, dataset):
    #Linear SVC with norm L1
    tuned_parameter =[{'C': [0.001 ,0.01, 0.1, 1, 10, 100, 1000]}]

    clf= GridSearchCV(LinearSVC(penalty= 'l1', dual= False), tuned_parameter)
    clf.fit(X_train, y_train)

    print("Best params:")
    print()
    print(clf.best_params_)
    print()

    # Besy value for C is 1.
    lsvc= LinearSVC(C= 1.0, penalty= 'l1', dual= False).fit(X_train, y_train)
    selector= SelectFromModel(estimator=lsvc, prefit= True)

    print("The feaures'coefficients that have to be removed", selector.get_support())
    print(lsvc.coef_)

    # 2. Feature Selection with SelectKBest (Univariate feature selection).

    # Applied the method for the 12 features to perform comparison among the corresponding ANOVA-F values

    best_features = SelectKBest(score_func=f_classif, k=12)
    fit = best_features.fit(X_train,y_train)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(dataset.columns)

    feature_scores = pd.concat([df_columns, df_scores],axis=1)
    feature_scores.columns = ['Feauture Name','Score'] 
    print(feature_scores.nlargest(12,'Score'))  

    # The features "time", "serum creatinine", "ejection_fraction", "serum sodium", "age" gained the highest F-values

    # 3. Method Random Forests.

    forest= RandomForestClassifier(n_estimators= 1000, random_state= 20)
    forest.fit(X_train, y_train)

    forest_feats= SelectFromModel(forest, threshold= 'median')
    forest_feats.fit(X_train, y_train)

    # Feature Importance.

    print(forest_feats.get_support(indices= True))
    importances = forest.feature_importances_
    plt.barh(np.arange(len(dataset.drop('DEATH_EVENT', axis=1).columns)), importances,align = 'center',tick_label = dataset.drop('DEATH_EVENT', axis=1).columns)
    plt.xlim((0,0.5))
    plt.grid(axis='x')
    plt.title('Importance of the features:')
    plt.show()

class Model_selection:
    def __init__(self, X_train, y_train, X_test, y_test, use_time=False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.use_time = use_time

        if use_time==True:
            self.X_train_new = X_train[:, [0, 4, 7, 8, 11]]
            self.X_test_new = X_test[:, [0, 4, 7, 8, 11]]
        else:
            self.X_train_new=X_train[:, [0, 4, 7, 8]]
            self.X_test_new = X_test[:, [0, 4, 7, 8]]

    def logistic_regression(self):
        lr = LogisticRegression(class_weight='balanced', random_state= 27)

        # Create the grid
        param_grid_12 = {"C" : np.arange(0,100,1),"penalty":["l1","l2"], "solver" : ['newton-cg', 'lbfgs', 'liblinear']}
        folds=10
        skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 999)

        # Apply gridsearch with 10 folds
        gridsearch_12 = GridSearchCV(estimator= lr, 
                                    param_grid= param_grid_12,
                                    cv=skf, 
                                    n_jobs=-1, 
                                    scoring='f1', 
                                    verbose=2).fit(self.X_train, self.y_train)

        print("GRIDSEARCH with LOGISTIC REGRESSION in the total number of features")
        print("Best parameters :")
        print(gridsearch_12.best_params_)
        print("f1-score :")
        print(gridsearch_12.best_score_)
        print("Best Estimator:")
        print(gridsearch_12.best_estimator_)
        print(gridsearch_12.best_estimator_.coef_)

        # Create the grid
        param_grid_5 = {"C" : np.arange(0,100,1),"penalty":["l1","l2"], "solver" : ['newton-cg', 'lbfgs', 'liblinear']}
        folds=10
        skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 999)

        gridsearch_5 = GridSearchCV(estimator= lr, 
                                    param_grid= param_grid_5,
                                    cv=skf, 
                                    n_jobs=-1, 
                                    scoring='f1', 
                                    verbose=2).fit(self.X_train_new, self.y_train)

        print("GRIDSEARCH with LOGISTIC REGRESSION in the 5 dominant features")
        print("Best parameters :")
        print(gridsearch_5.best_params_)
        print("f1-score :")
        print(gridsearch_5.best_score_)
        print("Best Estimator:")
        print(gridsearch_5.best_estimator_)
        print(gridsearch_5.best_estimator_.coef_)
        # WITH TIME
            # Best models when considering the total number of features
            # Best model: LogisticRegression(C=1, class_weight='balanced', random_state=27, solver='newton-cg') achieved f1-score: 0.749 
        # WITHOUT TIME
            # Best models when considering the dominant 4 features
            # Best model: LogisticRegression(C=6, class_weight='balanced', penalty='l1', random_state=27, solver='liblinear') achieved f1-score: 0.639.

    def random_forest(self):
        folds=10
        skf = StratifiedKFold(n_splits=folds, shuffle=True , random_state = 999)

        final_rf = RandomForestClassifier(class_weight='balanced', random_state=1234)
        gscv = GridSearchCV(estimator=final_rf,param_grid={
            "n_estimators":[50, 100, 500, 1000],
            "criterion":["gini","entropy"],
            "max_depth":[3,5,7],
            "min_samples_split":[80,100],
            "min_samples_leaf":[40,50],
        },cv=skf,n_jobs=-1,scoring="f1")

        gscv.fit(self.X_train,self.y_train)

        print("GRIDSEARCH with RANDOM FOREST in the total number of features")
        print("Best parameters :")
        print(gscv.best_params_)
        print("f1-score :")
        print(gscv.best_score_)
        print("Best Estimator:")
        print(gscv.best_estimator_)

        # Apply the algorithm in the 5 dominant features

        folds=10
        skf = StratifiedKFold(n_splits=folds, shuffle=True , random_state = 999)
        gscv_5 = GridSearchCV(estimator=final_rf,param_grid={
            "n_estimators":[50, 100, 500, 1000],
            "criterion":["gini","entropy"],
            "max_depth":[3,5,7],
            "min_samples_split":[80,100],
            "min_samples_leaf":[40,50],
        },cv=skf,n_jobs=-1,scoring="f1")

        gscv_5.fit(self.X_train_new,self.y_train)

        print("GRIDSEARCH with RANDOM FOREST in the 5 dominant features")
        print("Best parameters :")
        print(gscv_5.best_params_)
        print("f1-score :")
        print(gscv_5.best_score_)
        print("Best Estimator:")
        print(gscv_5.best_estimator_)
        # WITH TIME
            # Best models when considering the 5 dominant features
            # Best model: RandomForestClassifier(class_weight='balanced', criterion='gini', max_depth=3, min_samples_leaf=40, min_samples_split=80 ,random_state=1234) achieved f1-score: 0.761
        # WITHOUT TIME
            # Best models when considering the 4 dominant features
            # Best model: RandomForestClassifier(class_weight='balanced', max_depth=3, min_samples_leaf=40, min_samples_split=100, n_estimators=50, random_state=1234) achieved f1-score: 0.579

    def decision_tree(self):

        folds=10
        skf = StratifiedKFold(n_splits=folds, shuffle=True , random_state = 999)

        par_grid= {
            "max_depth": np.arange(1,10),
            "min_samples_split": [0.001, 0.01, 0.1, 0.2, 0.02, 0.002],
            "criterion": ["gini", "entropy"],
            "max_leaf_nodes": np.arange(1,10),
            'ccp_alpha': [0, 0.1, 0.001, 0.0001, 1]}


        clf_dt= DecisionTreeClassifier(class_weight='balanced', random_state=72)
        grid = GridSearchCV(clf_dt, par_grid, verbose=2, cv=skf,n_jobs=-1,scoring="f1")
        grid.fit(self.X_train, self.y_train)

        print("GRIDSEARCH with DECISION TREE in the total number of features")
        print("Best parameters :")
        print(grid.best_params_)
        print("f1-score :")
        print(grid.best_score_)
        print("Best Estimator:")
        print(grid.best_estimator_)

        grid_6= GridSearchCV(clf_dt, par_grid, verbose=2,cv=skf,n_jobs=-1,scoring="f1")
        grid_6.fit(self.X_train_new, self.y_train)

        print("GRIDSEARCH with DECISION TREE in the 5 dominant features")
        print("Best parameters :")
        print(grid_6.best_params_)
        print("f1-score :")
        print(grid_6.best_score_)
        print("Best Estimator:")
        print(grid_6.best_estimator_)
        # WITH TIME
            # Best models when considering the 5 dominant features
            # Best model: DecisionTreeClassifier(ccp_alpha=0, class_weight='balanced', criterion='entropy', max_depth=5, max_leaf_nodes=9, min_samples_split=0.1, random_state=72) achieved f1-score: 0.769.
        # WITHOUT TIME
            # Best models when considering the 4 dominant features
            # Best models: DecisionTreeClassifier(ccp_alpha=0, class_weight='balanced', criterion='gini', max_depth=6, max_leaf_nodes=7, min_samples_split=0.2, random_state=72)  με f1-score: 0.769.
    
    def visualize_best_tree(self):
        model_dt=DecisionTreeClassifier(ccp_alpha=0, class_weight='balanced',criterion='entropy', max_depth=5, max_leaf_nodes=9,min_samples_split=0.1, random_state=72)
        model_dt.fit(self.X_train_new, self.y_train)
        fig = plt.figure(figsize=(25,20))
        tree.plot_tree(model_dt, filled= True)
        plt.show()

    def svm(self):

        folds=10
        skf = StratifiedKFold(n_splits=folds, shuffle=True , random_state = 999)

        kernels = list(['linear', 'rbf', 'poly', 'sigmoid'])
        c = list([0.001, 0.01, 0.1, 1, 10, 100, 1000])
        gammas = list([0.1, 1, 10, 100])

        clf_svm = SVC(class_weight='balanced', random_state=17)
        param_grid_svm = dict(kernel=kernels, C=c, gamma=gammas)
        grid_svm = GridSearchCV(clf_svm, param_grid_svm, cv=skf, n_jobs=-1, scoring='f1')
        grid_svm.fit(self.X_train, self.y_train)

        print("GRIDSEARCH with SVM in the total number of features")
        print("Best parameters :")
        print(grid_svm.best_params_)
        print("f1-score :")
        print(grid_svm.best_score_)
        print("Best Estimator:")
        print(grid_svm.best_estimator_)
        # WITH TIME
            # Best model: SVC(C=1, class_weight='balanced', gamma=0.1, kernel='linear',random_state=17) achieved f1-score: 0.723.
        # WITHOUT TIME
            # Best model: SVC(C=1000, class_weight='balanced', gamma=0.1, kernel='linear',random_state=17) achieved f1-score: 0.603.

    def evaluation(self):
        if self.use_time==True:
            # LOGISTIC REGRESSION
            clf_1=LogisticRegression(C=1, class_weight='balanced', random_state=27,solver='newton-cg')
            clf_1.fit(self.X_train, self.y_train)
            y_pred_1=clf_1.predict(self.X_test)
            y_score_1 = clf_1.decision_function(self.X_test)
            average_precision_1 = average_precision_score(self.y_test, y_score_1)

            # RANDOM FOREST 
            clf_2=RandomForestClassifier(class_weight='balanced', criterion='gini',max_depth=3,min_samples_leaf=40, min_samples_split=80, random_state=123456)
            clf_2.fit(self.X_train_new, self.y_train)
            y_pred_2=clf_2.predict(self.X_test_new)

            # DECISION TREES
            clf_3=DecisionTreeClassifier(ccp_alpha=0, class_weight='balanced',criterion='entropy', max_depth=5, max_leaf_nodes=9,min_samples_split=0.1, random_state=72)
            clf_3.fit(self.X_train_new, self.y_train)
            y_pred_3=clf_3.predict(self.X_test_new)

            # SVM
            clf_4=SVC(C=1, class_weight='balanced', gamma=0.1, kernel='linear', random_state=17)
            clf_4.fit(self.X_train, self.y_train)
            y_pred_4=clf_4.predict(self.X_test)
            y_score_4 = clf_4.decision_function(self.X_test)
            average_precision_4 = average_precision_score(self.y_test, y_score_4)

            df1 = pd.DataFrame(
            {
            "Accuracy": [accuracy_score(self.y_test, y_pred_1), accuracy_score(self.y_test, y_pred_2), accuracy_score(self.y_test, y_pred_3), accuracy_score(self.y_test, y_pred_4)],
            "F1-score": [f1_score(self.y_test, y_pred_1),f1_score(self.y_test, y_pred_2), f1_score(self.y_test, y_pred_3) ,f1_score(self.y_test, y_pred_4) ],
            "Precision": [precision_score(self.y_test, y_pred_1), precision_score(self.y_test, y_pred_2), precision_score(self.y_test, y_pred_3), precision_score(self.y_test, y_pred_4)],
            "Recall": [recall_score(self.y_test, y_pred_1), recall_score(self.y_test, y_pred_2), recall_score(self.y_test, y_pred_3), recall_score(self.y_test, y_pred_4)],
            },index=['Logistic Regression', 'Random Forest', 'Decision Tree', 'SVM'],)

            print('Average precision-recall score for Logistic Regression:', average_precision_1)
            print('Average precision-recall score for SVM:', average_precision_4)
            print('Confusion matrix for Logistic Regression:', confusion_matrix(self.y_test, y_pred_1))
            print('Confusion matrix for Random Forest:', confusion_matrix(self.y_test, y_pred_2))
            print('Confusion matrix for Decision Tree:', confusion_matrix(self.y_test, y_pred_3))
            print('Confusion matrix for SVM:', confusion_matrix(self.y_test, y_pred_4))

            disp_1 = plot_precision_recall_curve(clf_1, self.X_test, self.y_test)
            disp_2 = plot_precision_recall_curve(clf_2, self.X_test_new, self.y_test)
            disp_3 = plot_precision_recall_curve(clf_3, self.X_test_new, self.y_test)
            disp_4 = plot_precision_recall_curve(clf_4, self.X_test, self.y_test)

            return df1
        else:
            # LOGISTIC REGRESSION
            clf_1=LogisticRegression(C=6, class_weight='balanced', penalty='l1', random_state=27,solver='liblinear')
            clf_1.fit(self.X_train_new, self.y_train)
            y_pred_1=clf_1.predict(self.X_test_new)
            y_score_1 = clf_1.decision_function(self.X_test_new)
            average_precision_1 = average_precision_score(self.y_test, y_score_1)

            # RANDOMFOREST
            clf_2=RandomForestClassifier(class_weight='balanced', max_depth=3,min_samples_leaf=40, min_samples_split=100, n_estimators=50, random_state=123456)
            clf_2.fit(self.X_train_new, self.y_train)
            y_pred_2=clf_2.predict(self.X_test_new)

            # DECISION TREES
            clf_3=DecisionTreeClassifier(ccp_alpha=0, class_weight='balanced', max_depth=6, max_leaf_nodes=7, min_samples_split=0.2, random_state=72)
            clf_3.fit(self.X_train_new, self.y_train)
            y_pred_3=clf_3.predict(self.X_test_new)

            # SVM
            clf_4=SVC(C=1000, class_weight='balanced', gamma=0.1, kernel='linear', random_state=17)
            clf_4.fit(self.X_train, self.y_train)
            y_pred_4=clf_4.predict(self.X_test)
            y_score_4 = clf_4.decision_function(self.X_test)
            average_precision_4 = average_precision_score(self.y_test, y_score_4)


            df2 = pd.DataFrame(
            {
            "Accuracy": [accuracy_score(self.y_test, y_pred_1), accuracy_score(self.y_test, y_pred_2), accuracy_score(self.y_test, y_pred_3), accuracy_score(self.y_test, y_pred_4)],
            "F1-score": [f1_score(self.y_test, y_pred_1),f1_score(self.y_test, y_pred_2), f1_score(self.y_test, y_pred_3) ,f1_score(self.y_test, y_pred_4) ],
            "Precision": [precision_score(self.y_test, y_pred_1), precision_score(self.y_test, y_pred_2), precision_score(self.y_test, y_pred_3), precision_score(self.y_test, y_pred_4)],
            "Recall": [recall_score(self.y_test, y_pred_1), recall_score(self.y_test, y_pred_2), recall_score(self.y_test, y_pred_3), recall_score(self.y_test, y_pred_4)],
            },index=['Logistic Regression', 'Random Forest', 'Decision Tree', 'SVM'],)

            print('Average precision-recall score for Logistic Regression:', average_precision_1)
            print('Average precision-recall score for SVM:', average_precision_4)
            print('Confusion matrix for  confusion_matrix for Logistic Regression:', confusion_matrix(self.y_test, y_pred_1))
            print('Confusion matrix for  confusion_matrix for Random Forest:', confusion_matrix(self.y_test, y_pred_2))
            print('Confusion matrix for  confusion_matrix for Decision Tree:', confusion_matrix(self.y_test, y_pred_3))
            print('Confusion matrix for  confusion_matrix for SVM:', confusion_matrix(self.y_test, y_pred_4))


            disp_1 = plot_precision_recall_curve(clf_1, self.X_test_new, self.y_test)
            disp_2 = plot_precision_recall_curve(clf_2, self.X_test_new, self.y_test)
            disp_3 = plot_precision_recall_curve(clf_3, self.X_test_new, self.y_test)
            disp_4 = plot_precision_recall_curve(clf_4, self.X_test, self.y_test)

            return df2
