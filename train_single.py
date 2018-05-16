# -*- coding:utf-8 -*-
if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns;sns.set()
    import re
    from sklearn.svm import *
    from sklearn.preprocessing import *
    from sklearn.pipeline import *
    from sklearn.model_selection import *
    from sklearn.ensemble import *
    # 正式流程
    train_df_org = pd.read_csv('data/train.csv')
    test_df_org = pd.read_csv('data/test.csv')
    test_df_org['Survived'] = 0
    combined_train_test = pd.concat([train_df_org, test_df_org], ignore_index=True)
    PassengerId = test_df_org['PassengerId']

    # 数据预处理
    #    Embarked
    combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().values[0], inplace = True)
    combined_train_test['Embarked'] = pd.factorize(combined_train_test['Embarked'])[0]
    emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'], prefix='Embarked')
    combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis='columns')

    #    Sex
    combined_train_test['Sex'] = pd.factorize(combined_train_test['Sex'])[0]
    sex_dummines_df = pd.get_dummies(combined_train_test['Sex'], prefix='Sex')
    combined_train_test = pd.concat([combined_train_test, sex_dummines_df], axis='columns')

    #    Name
    combined_train_test['Title'] = combined_train_test['Name'].map(lambda x: re.compile(', (.*?)\.').findall(x)[0])
    title_Dict = {}
    title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
    title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
    title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
    title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
    title_Dict.update(dict.fromkeys(['Sir','Mr'], 'Mr'))
    title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
    combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict)

    combined_train_test['Title'] = pd.factorize(combined_train_test['Title'])[0]
    title_dummies_df = pd.get_dummies(combined_train_test['Title'], prefix='Title')
    combined_train_test = pd.concat([combined_train_test, title_dummies_df], axis=1)
    combined_train_test['Name_length'] = combined_train_test['Name'].map(len)

    #    Fare
    combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(
        combined_train_test.groupby('Pclass').transform(np.mean))
    combined_train_test['Group_Ticket'] = combined_train_test.groupby('Ticket')['Fare'].transform('count')
    combined_train_test['Fare'] = combined_train_test['Fare'] / combined_train_test['Group_Ticket']
    combined_train_test.drop(['Group_Ticket'], axis=1, inplace=True)

    combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'], 5)
    combined_train_test['Fare_bin_id'] = pd.factorize(combined_train_test['Fare_bin'])[0]
    fare_bin_dummies_id = pd.get_dummies(combined_train_test['Fare_bin_id'], prefix='Fare')
    combined_train_test = pd.concat([combined_train_test, fare_bin_dummies_id], axis=1).drop('Fare_bin', axis=1)

    #    按照价格对同一个Pclass进行二分
    Pclass_mean_fare = combined_train_test.groupby(['Pclass'])['Fare'].mean()
    combined_train_test['Pclass_Fare_Category'] = combined_train_test.apply(
        lambda x: 'Pclass' + str(x['Pclass']) + '_Low'
        if x['Fare'] < Pclass_mean_fare[x['Pclass']] else 'Pclass' + str(x['Pclass']) + '_High', axis=1)

    from sklearn.preprocessing import LabelEncoder

    pclass_level = LabelEncoder()
    pclass_level.fit(
        np.array(['Pclass1_Low', 'Pclass1_High', 'Pclass2_Low', 'Pclass2_High', 'Pclass3_Low', 'Pclass3_High']))

    combined_train_test['Pclass_Fare_Category'] = pclass_level.transform(combined_train_test['Pclass_Fare_Category'])

    pclass_dummies_df = pd.get_dummies(combined_train_test['Pclass_Fare_Category'], prefix='Pclass')
    combined_train_test = pd.concat([combined_train_test, pclass_dummies_df], axis=1)

    #    Pclass 特征化
    combined_train_test['Pclass'] = pd.factorize(combined_train_test['Pclass'])[0]

    #    Parch and SibSp
    combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1


    def family_size_category(family_size):
        if family_size <= 1:
            return 'Signal'
        elif family_size <= 4:
            return 'Small_Family'
        else:
            return 'Big_Family'


    combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].apply(family_size_category)
    le_family = LabelEncoder()
    le_family.fit(np.array(['Signal', 'Small_Family', 'Big_Family']))
    combined_train_test['Family_Size_Category'] = le_family.transform(combined_train_test['Family_Size_Category'])

    family_size_dummies_df = pd.get_dummies(combined_train_test['Family_Size_Category'], prefix='Family_Size_Category')
    combined_train_test = pd.concat([combined_train_test, family_size_dummies_df], axis=1)

    #    Age

    missing_age_df = pd.DataFrame(combined_train_test[[
        'Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category', 'Fare', 'Fare_bin_id',
        'Pclass']])
    missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
    missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor


    def fill_missing_age(missing_age_train, missing_age_test):
        missing_age_train_X = missing_age_train.drop(['Age'], axis=1)
        missing_age_train_Y = missing_age_train['Age']
        missing_age_test_X = missing_age_test.drop(['Age'], axis=1)

        # mode 1 gbm
        gbm_reg = GradientBoostingRegressor(random_state=42)
        gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [8], 'learning_rate': [0.01]}
        gbm_reg_grid = GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, verbose=1,
                                                    scoring='neg_mean_squared_error')
        gbm_reg_grid.fit(missing_age_train_X, missing_age_train_Y)
        print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
        print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
        print('GB Train Error for "Age" Feature Regressor:' + str(
            gbm_reg_grid.score(missing_age_train_X, missing_age_train_Y)))
        missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(missing_age_test_X)
        print(missing_age_test['Age_GB'][:4])

        # mode 2 rf
        rf_reg = GradientBoostingRegressor(random_state=42)
        rf_reg_param_grid = {'n_estimators': [2000], 'max_depth': [5], 'random_state': [0]}
        rf_reg_grid = GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, verbose=1,
                                                   scoring='neg_mean_squared_error')
        rf_reg_grid.fit(missing_age_train_X, missing_age_train_Y)
        print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))
        print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))
        print('RF Train Error for "Age" Feature Regressor:' + str(
            rf_reg_grid.score(missing_age_train_X, missing_age_train_Y)))
        missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_test_X)
        print(missing_age_test['Age_RF'][:4])

        # two models merge
        print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_RF']].mode(axis=1).shape)

        missing_age_test.loc[:, 'Age'] = (missing_age_test['Age_GB'] + missing_age_test['Age_RF']) / 2
        print(missing_age_test['Age'][:4])

        missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)
        return missing_age_test


    combined_train_test.loc[(combined_train_test['Age'].isnull()), 'Age'] = fill_missing_age(missing_age_train,
                                                                                             missing_age_test)

    #   Ticket
    combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0].apply(
        lambda x: 'U0' if x.isnumeric() else x)
    combined_train_test['Ticket_Letter'] = pd.factorize(combined_train_test['Ticket_Letter'])[0]

    #   Cabin
    combined_train_test['Cabin'].fillna('U0', inplace=True)
    combined_train_test['Cabin'] = combined_train_test['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)

    # 特征间相关性分析
    Correlation = pd.DataFrame(combined_train_test[['Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size',
                                                    'Family_Size_Category', 'Fare', 'Fare_bin_id', 'Pclass',
                                                    'Pclass_Fare_Category', 'Age', 'Ticket_Letter', 'Cabin']])

    # 输入模型前的一些处理
    #    1. 一些数据的正则化 —— 正则化应在管道内进行，否则会污染数据
    # from sklearn import preprocessing
    #
    # scale_age_fare = preprocessing.StandardScaler().fit(combined_train_test[['Age', 'Fare', 'Name_length']])
    # combined_train_test[['Age', 'Fare', 'Name_length']] = scale_age_fare.transform(
    #     combined_train_test[['Age', 'Fare', 'Name_length']])

    #    2. 弃掉无用特征
    combined_data_backup = combined_train_test
    combined_train_test.drop(['PassengerId', 'Embarked', 'Sex', 'Name', 'Title', 'Fare_bin_id', 'Pclass_Fare_Category',
                              'Parch', 'SibSp', 'Family_Size_Category', 'Ticket'], axis=1, inplace=True)

    #    3. 将训练数据和测试数据分开：
    titanic_train_data_X = combined_train_test[:891].drop(['Survived'], axis=1)
    titanic_train_data_Y = combined_train_test[:891]['Survived']
    titanic_test_data_X = combined_train_test[891:].drop(['Survived'], axis=1)

    # 模型融合及测试

    #    (1) 利用不同的模型来对特征进行筛选，选出较为重要的特征：

    #    (2) 依据我们筛选出的特征构建训练集和测试集
    X_train,X_test,y_train,y_test = train_test_split(titanic_train_data_X,titanic_train_data_Y,random_state=0)

    #管道搜索模型参数
    from sklearn.ensemble import RandomForestClassifier
    pipe = Pipeline([('preprocessing',StandardScaler()),('classifier',SVC())])
    param_grid = [{'classifier':[SVC()],'preprocessing':[StandardScaler()],
                    'classifier__C':[0.1,1],'classifier__gamma':[0.1,1],'classifier__kernel':['poly']},
                   {'preprocessing':[None],'classifier':[RandomForestClassifier()],
                    'classifier__n_estimators':[300,500],'classifier__max_depth':[3,5],'classifier__max_features':['log2'],
                    'classifier__min_samples_leaf':[3,5]}]
    grid = GridSearchCV(pipe, param_grid, cv = 5, n_jobs = -1, verbose = 2)
    grid.fit(X_train,y_train)
    print('score :{}'.format(grid.score(X_test,y_test)))

    from sklearn.feature_selection import RFE
    select = RFE(RandomForestClassifier(n_estimators=300,max_depth=3,max_features='log2',min_samples_leaf=3))
    select.fit(X_train,y_train)
    X_train_rfe = select.transform(X_train)
    X_test_rfe = select.transform(X_test)
    grid.fit(X_train_rfe,y_train)
    print('score :{}'.format(grid.score(X_test_rfe,y_test)))

    '''
    x_train = titanic_train_data_X.values  # Creates an array of the train data
    x_test = titanic_test_data_X.values  # Creats an array of the test data
    y_train = titanic_train_data_Y.values

    from sklearn.model_selection import KFold

    # Some useful parameters which will come in handy later on
    ntrain = titanic_train_data_X.shape[0]
    ntest = titanic_test_data_X.shape[0]
    SEED = 0  # for reproducibility
    NFOLDS = 7  # set folds for out-of-fold prediction
    kf = KFold(n_splits=NFOLDS, random_state=SEED, shuffle=False)


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

    rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test) # Random Forest
    svm_oof_train, svm_oof_test = get_out_fold(svc, x_train, y_train, x_test) # Support Vector

    print("Training is complete")

    #    (4) 预测并生成提交文件
    #    Level 2：XGBoost

    x_train = np.concatenate((rf_oof_train, svm_oof_train), axis=1)
    x_test = np.concatenate((rf_oof_test,  svm_oof_test), axis=1)

    from xgboost import XGBClassifier

    gbm = XGBClassifier(n_estimators=300,gamma=0.1,max_depth=8,min_child_weight=3,subsample=0.8,colsample_bytree=0.8,reg_lambda=0.1,reg_alpha=0.5)
    #gbm_param = {'n_estimators':[300],'gamma':[0.1], 'max_depth':[8], 'min_child_weight':[3],
    #             'subsample':[0.8], 'colsample_bytree':[0.8], 'reg_lambda':[0.1], 'reg_alpha':[0.5]}
    #gbm = model_fit(gbm,gbm_param,titanic_train_data_X, titanic_train_data_Y,True)
    gbm.fit(x_train,y_train)
    predictions = gbm.predict(x_test)

    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
    StackingSubmission.to_csv('StackingSubmission.csv',index=False,sep=',')


    '''
    #    (4) 预测并生成提交文件
    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': grid.predict(titanic_test_data_X)})
    StackingSubmission.to_csv('StackingSubmission.csv', index=False, sep=',')
