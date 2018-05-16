# -*- coding:utf-8 -*-
if __name__ == '__main__':
    '''
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    import seaborn as sns
    import re
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    
    #缺失数据处理
    train_data['Embarked'] =train_data['Embarked'].fillna(train_data['Embarked'].dropna().mode().values[0])
    train_data['Cabin'] = train_data['Cabin'].fillna('U0')
    
    age_df = train_data[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
    age_df_notnull = age_df.loc[train_data['Age'].notnull()]
    age_df_isnull = age_df.loc[train_data['Age'].isnull()]
    
    X = age_df_notnull.values[:,1:]
    Y = age_df_notnull.values[:,0]
    
    RFR = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
    RFR.fit(X,Y)
    predictAge = RFR.predict(age_df_isnull.values[:,1:])
    train_data.loc[train_data['Age'].isnull(),['Age']] = predictAge
    '''

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import re
    #正式流程
    train_df_org = pd.read_csv('data/train.csv')
    test_df_org = pd.read_csv('data/test.csv')
    test_df_org['Survived'] = 0
    combined_train_test = pd.concat([train_df_org,test_df_org],ignore_index=True)
    PassengerId = test_df_org['PassengerId']

    #数据预处理
    #    Embarked
    combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().values[0], inplace = True)
    combined_train_test['Embarked'] = pd.factorize(combined_train_test['Embarked'])[0]
    emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'],prefix = 'Embarked')
    combined_train_test = pd.concat([combined_train_test,emb_dummies_df],axis = 'columns')

    #    Sex
    combined_train_test['Sex'] = pd.factorize(combined_train_test['Sex'])[0]
    sex_dummines_df = pd.get_dummies(combined_train_test['Sex'],prefix = 'Sex')
    combined_train_test = pd.concat([combined_train_test,sex_dummines_df],axis = 'columns')

    #    Name
    combined_train_test['Title'] = combined_train_test['Name'].map(lambda x: re.compile(', (.*?)\.').findall(x)[0])
    title_Dict = {}
    title_Dict.update(dict.fromkeys(['Capt','Col','Major','Dr','Rev'],'Officer'))
    title_Dict.update(dict.fromkeys(['Don','Sir','the Countess','Dona','Lady'],'Royalty'))
    title_Dict.update(dict.fromkeys(['Mme','Ms','Mrs'],'Mrs'))
    title_Dict.update(dict.fromkeys(['Mlle','Mill'],'Miss'))
    title_Dict.update(dict.fromkeys(['Mr'],'Mr'))
    title_Dict.update(dict.fromkeys(['Master','Jonkher'],'Master'))
    combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict)

    combined_train_test['Title'] = pd.factorize(combined_train_test['Title'])[0]
    title_dummies_df = pd.get_dummies(combined_train_test['Title'],prefix = 'Title')
    combined_train_test = pd.concat([combined_train_test,title_dummies_df], axis = 1)
    combined_train_test['Name_length'] = combined_train_test['Name'].map(len)

    #    Fare
    combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform(np.mean))
    combined_train_test['Group_Ticket'] = combined_train_test.groupby('Ticket')['Fare'].transform('count')
    combined_train_test['Fare'] = combined_train_test['Fare']/combined_train_test['Group_Ticket']
    combined_train_test.drop(['Group_Ticket'], axis = 1, inplace = True)

    combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'],5)
    combined_train_test['Fare_bin_id'] = pd.factorize(combined_train_test['Fare_bin'])[0]
    fare_bin_dummies_id = pd.get_dummies(combined_train_test['Fare_bin_id'],prefix='Fare')
    combined_train_test = pd.concat([combined_train_test,fare_bin_dummies_id],axis = 1).drop('Fare_bin',axis = 1)

    #    按照价格对同一个Pclass进行二分
    Pclass_mean_fare = combined_train_test.groupby(['Pclass'])['Fare'].mean()
    combined_train_test['Pclass_Fare_Category'] = combined_train_test.apply(
        lambda x : 'Pclass' + str(x['Pclass']) + '_Low'
        if x['Fare'] < Pclass_mean_fare[x['Pclass']] else 'Pclass' + str(x['Pclass']) + '_High',axis = 1 )

    from sklearn.preprocessing import LabelEncoder

    pclass_level = LabelEncoder()
    pclass_level.fit(np.array(['Pclass1_Low','Pclass1_High','Pclass2_Low','Pclass2_High','Pclass3_Low','Pclass3_High']))

    combined_train_test['Pclass_Fare_Category'] = pclass_level.transform(combined_train_test['Pclass_Fare_Category'])

    pclass_dummies_df = pd.get_dummies(combined_train_test['Pclass_Fare_Category'],prefix ='Pclass')
    combined_train_test = pd.concat([combined_train_test,pclass_dummies_df],axis = 1)

    #    Pclass 特征化
    combined_train_test['Pclass'] = pd.factorize(combined_train_test['Pclass'])[0]

    #    Parch and SibSp
    combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
    def family_size_category(family_size):
        if family_size <= 1:
            return 'Signal'
        elif family_size <=4:
            return 'Small_Family'
        else :
            return 'Big_Family'

    combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].apply(family_size_category)
    le_family = LabelEncoder()
    le_family.fit(np.array(['Signal','Small_Family','Big_Family']))
    combined_train_test['Family_Size_Category'] = le_family.transform(combined_train_test['Family_Size_Category'])

    family_size_dummies_df = pd.get_dummies(combined_train_test['Family_Size_Category'],prefix = 'Family_Size_Category')
    combined_train_test = pd.concat([combined_train_test,family_size_dummies_df],axis = 1)

    #    Age

    missing_age_df = pd.DataFrame(combined_train_test[[
        'Age','Embarked','Sex','Title','Name_length','Family_Size','Family_Size_Category','Fare','Fare_bin_id','Pclass']])
    missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
    missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]

    from sklearn import ensemble
    from sklearn import model_selection
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor

    def fill_missing_age(missing_age_train,missing_age_test):
        missing_age_train_X = missing_age_train.drop(['Age'],axis = 1)
        missing_age_train_Y = missing_age_train['Age']
        missing_age_test_X = missing_age_test.drop(['Age'],axis = 1)

        #mode 1 gbm
        gbm_reg = GradientBoostingRegressor(random_state=42)
        gbm_reg_param_grid = {'n_estimators':[2000], 'max_depth':[4],'learning_rate':[0.01]}
        gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv = 10, verbose = 1,scoring='neg_mean_squared_error')
        gbm_reg_grid.fit(missing_age_train_X,missing_age_train_Y)
        print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
        print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
        print('GB Train Error for "Age" Feature Regressor:' + str(gbm_reg_grid.score(missing_age_train_X,missing_age_train_Y)))
        missing_age_test.loc[:,'Age_GB'] = gbm_reg_grid.predict(missing_age_test_X)
        print(missing_age_test['Age_GB'][:4])

        #mode 2 rf
        rf_reg = GradientBoostingRegressor(random_state=42)
        rf_reg_param_grid = {'n_estimators':[2000], 'max_depth':[5],'random_state':[0]}
        rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv = 10, verbose = 1,scoring='neg_mean_squared_error')
        rf_reg_grid.fit(missing_age_train_X,missing_age_train_Y)
        print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))
        print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))
        print('RF Train Error for "Age" Feature Regressor:' + str(rf_reg_grid.score(missing_age_train_X,missing_age_train_Y)))
        missing_age_test.loc[:,'Age_RF'] = rf_reg_grid.predict(missing_age_test_X)
        print(missing_age_test['Age_RF'][:4])

        #two models merge
        print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB','Age_RF']].mode(axis=1).shape)

        missing_age_test.loc[:,'Age'] = (missing_age_test['Age_GB'] + missing_age_test['Age_RF'])/2
        print(missing_age_test['Age'][:4])

        missing_age_test.drop(['Age_GB','Age_RF'],axis = 1,inplace = True)
        return missing_age_test

    combined_train_test.loc[(combined_train_test['Age'].isnull()),'Age'] = fill_missing_age(missing_age_train,missing_age_test)

    #   Ticket
    combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0].apply(lambda x :'U0' if x.isnumeric() else x)
    combined_train_test['Ticket_Letter'] = pd.factorize(combined_train_test['Ticket_Letter'])[0]

    #   Cabin
    combined_train_test['Cabin'].fillna('U0',inplace=True)
    combined_train_test['Cabin'] = combined_train_test['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)

    #特征间相关性分析
    Correlation = pd.DataFrame(combined_train_test[['Embarked','Sex','Title','Name_length','Family_Size',
                                                    'Family_Size_Category','Fare','Fare_bin_id','Pclass',
                                                    'Pclass_Fare_Category','Age','Ticket_Letter','Cabin']])

    #输入模型前的一些处理
    #    1. 一些数据的正则化
    from sklearn import preprocessing
    scale_age_fare = preprocessing.StandardScaler().fit(combined_train_test[['Age','Fare', 'Name_length']])
    combined_train_test[['Age','Fare', 'Name_length']] = scale_age_fare.transform(combined_train_test[['Age','Fare', 'Name_length']])

    #    2. 弃掉无用特征
    combined_data_backup = combined_train_test
    combined_train_test.drop(['PassengerId', 'Embarked', 'Sex', 'Name', 'Title', 'Fare_bin_id', 'Pclass_Fare_Category',
                           'Parch', 'SibSp', 'Family_Size_Category', 'Ticket'],axis=1,inplace=True)

    #    3. 将训练数据和测试数据分开：
    train_data = combined_train_test[:891]
    test_data = combined_train_test[891:]

    titanic_train_data_X = train_data.drop(['Survived'],axis=1)
    titanic_train_data_Y = train_data['Survived']
    titanic_test_data_X = test_data.drop(['Survived'],axis=1)

    #模型融合及测试

    #     (1) 利用不同的模型来对特征进行筛选，选出较为重要的特征：
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier


    def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):
        # random forest
        rf_est = RandomForestClassifier(random_state=0)
        rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
        rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=-1, cv=10, verbose=1)
        rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)
        print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
        print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
        print('Top N Features RF Train Score:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))
        feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),
                                              'importance': rf_grid.best_estimator_.feature_importances_}).sort_values(
            'importance', ascending=False)
        features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
        print('Sample 10 Features from RF Classifier')
        print(str(features_top_n_rf[:10]))

        # AdaBoost
        ada_est = AdaBoostClassifier(random_state=0)
        ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
        ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=-1, cv=10, verbose=1)
        ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
        print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
        print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
        print('Top N Features Ada Train Score:' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))
        feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
                                               'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
            'importance', ascending=False)
        features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
        print('Sample 10 Feature from Ada Classifier:')
        print(str(features_top_n_ada[:10]))

        # ExtraTree
        et_est = ExtraTreesClassifier(random_state=0)
        et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
        et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=-1, cv=10, verbose=1)
        et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
        print('Top N Features Best ET Params:' + str(et_grid.best_params_))
        print('Top N Features Best ET Score:' + str(et_grid.best_score_))
        print('Top N Features ET Train Score:' + str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))
        feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),
                                              'importance': et_grid.best_estimator_.feature_importances_}).sort_values(
            'importance', ascending=False)
        features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
        print('Sample 10 Features from ET Classifier:')
        print(str(features_top_n_et[:10]))

        # GradientBoosting
        gb_est = GradientBoostingClassifier(random_state=0)
        gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
        gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=-1, cv=10, verbose=1)
        gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)
        print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
        print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
        print('Top N Features GB Train Score:' + str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))
        feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),
                                              'importance': gb_grid.best_estimator_.feature_importances_}).sort_values(
            'importance', ascending=False)
        features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
        print('Sample 10 Feature from GB Classifier:')
        print(str(features_top_n_gb[:10]))

        # DecisionTree
        dt_est = DecisionTreeClassifier(random_state=0)
        dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
        dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=-1, cv=10, verbose=1)
        dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)
        print('Top N Features Best DT Params:' + str(dt_grid.best_params_))
        print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
        print('Top N Features DT Train Score:' + str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))
        feature_imp_sorted_dt = pd.DataFrame({'feature': list(titanic_train_data_X),
                                              'importance': dt_grid.best_estimator_.feature_importances_}).sort_values(
            'importance', ascending=False)
        features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
        print('Sample 10 Features from DT Classifier:')
        print(str(features_top_n_dt[:10]))

        # merge the three models
        features_top_n = pd.concat(
            [features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt],
            ignore_index=True).drop_duplicates()

        features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et,
                                         feature_imp_sorted_gb, feature_imp_sorted_dt], ignore_index=True)

        return features_top_n, features_importance

    #    (2) 依据我们筛选出的特征构建训练集和测试集
    feature_to_pick = 30
    feature_top_n, feature_importance = get_top_n_features(titanic_train_data_X, titanic_train_data_Y, feature_to_pick)
    titanic_train_data_X = pd.DataFrame(titanic_train_data_X[feature_top_n])
    titanic_test_data_X = pd.DataFrame(titanic_test_data_X[feature_top_n])

    '''
    # 特征值可视化
    rf_feature_imp = feature_importance[:10]
    feature_len = int(len(feature_importance)/5)
    Ada_feature_imp = feature_importance[feature_len:feature_len+10].reset_index(drop=True)
    
    # make importances relative to max importance
    rf_feature_importance = 100.0 * (rf_feature_imp['importance'] / rf_feature_imp['importance'].max())
    Ada_feature_importance = 100.0 * (Ada_feature_imp['importance'] / Ada_feature_imp['importance'].max())
    
    # Get the indexes of all features over the importance threshold
    rf_important_idx = np.where(rf_feature_importance)[0]
    Ada_important_idx = np.where(Ada_feature_importance)[0]
    
    # Adapted from Gradient Boosting regression
    pos = np.arange(rf_important_idx.shape[0]) + .5
    
    plt.figure(1, figsize = (18, 8))
    
    plt.subplot(121)
    plt.barh(pos, rf_feature_importance[rf_important_idx][::-1])
    plt.yticks(pos, rf_feature_imp['feature'][::-1])
    plt.xlabel('Relative Importance')
    plt.title('RandomForest Feature Importance')
    
    plt.subplot(122)
    plt.barh(pos, Ada_feature_importance[Ada_important_idx][::-1])
    plt.yticks(pos, Ada_feature_imp['feature'][::-1])
    plt.xlabel('Relative Importance')
    plt.title('AdaBoost Feature Importance')
    
    plt.show()
    '''
    #    (3) 模型融合（Model Ensemble）
    #     Stacking框架融合:
    #           Level 1：RandomForest、AdaBoost、ExtraTrees、GBDT、DecisionTree、KNN、SVM
    from sklearn.model_selection import KFold

    # Some useful parameters which will come in handy later on
    ntrain = titanic_train_data_X.shape[0]
    ntest = titanic_test_data_X.shape[0]
    SEED = 0 # for reproducibility
    NFOLDS = 7 # set folds for out-of-fold prediction
    kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=False)

    def get_out_fold(clf, x_train, y_train, x_test):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf.split(x_train)):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]

            clf.fit(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt',max_depth=6,
                                min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)

    ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)

    et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)

    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)

    dt = DecisionTreeClassifier(max_depth=8)

    knn = KNeighborsClassifier(n_neighbors = 2)

    svm = SVC(kernel='linear', C=0.025)

    # Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
    x_train = titanic_train_data_X.values # Creates an array of the train data
    x_test = titanic_test_data_X.values # Creats an array of the test data
    y_train = titanic_train_data_Y.values

    # Create our OOF train and test predictions. These base results will be used as new features
    rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test) # Random Forest
    ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_test) # AdaBoost
    et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test) # Extra Trees
    gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test) # Gradient Boost
    dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_test) # Decision Tree
    knn_oof_train, knn_oof_test = get_out_fold(knn, x_train, y_train, x_test) # KNeighbors
    svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test) # Support Vector

    print("Training is complete")

    #    (4) 预测并生成提交文件
    #    Level 2：XGBoost

    x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, knn_oof_train, svm_oof_train), axis=1)
    x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, knn_oof_test, svm_oof_test), axis=1)

    from xgboost import XGBClassifier

    gbm = XGBClassifier( n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8,
                         colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1).fit(x_train, y_train)
    predictions = gbm.predict(x_test)

    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
    StackingSubmission.to_csv('StackingSubmission.csv',index=False,sep=',')

