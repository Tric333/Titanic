# -*- coding:utf-8 -*-
PLOT_LEARN_CURVE = False
if __name__ == '__main__':
    import time
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns;sns.set()
    import re

    #model validation
    from sklearn.model_selection import *
    from sklearn.metrics import *
    from sklearn.metrics import roc_auc_score

    #model compose
    from sklearn.pipeline import *
    from sklearn.pipeline import make_pipeline

    # data process
    from sklearn.preprocessing import *
    from sklearn.feature_selection import RFE
    from sklearn.decomposition import *
    from data_preprocess import *

    # model import
    from model.svc_model import *

    # 正式流程 - load data
    train_df_org = pd.read_csv('data/train.csv')
    test_df_org = pd.read_csv('data/test.csv')
    test_df_org['Survived'] = 0
    combined_train_test = pd.concat([train_df_org, test_df_org], ignore_index=True)
    PassengerId = test_df_org['PassengerId']


    # 数据预处理
    #    Embarked
    combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().values[0], inplace = True)
    #    Name
    combined_train_test['Title'] = combined_train_test['Name'].map(lambda x: re.compile(', (.*?)\.').findall(x)[0])
    #    Name_length
    combined_train_test['Name_length'] = combined_train_test['Name'].map(len)
    #    Fare
    combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(
        combined_train_test.groupby('Pclass').transform(np.mean))
    #    Parch and SibSp
    #    Family Size 分为1个人, 2-4人，4人以上
    combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
    combined_train_test['Family_Size_Category'] = pd.cut(combined_train_test.Family_Size,[0,1,4,100])

    # Group Ticket 团体生存率堪忧 单独拿出来
    # 团体票 + 高 SibSp 生存率更低 后续添加
    combined_train_test['Group_Ticket'] = combined_train_test.groupby('Ticket')['Fare'].transform('count')
    combined_train_test['Fare'] = combined_train_test['Fare'] / combined_train_test['Group_Ticket']

    #    fare_bin
    combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'], 5)

    #   Ticket
    combined_train_test['Ticket_Letter'] = combined_train_test['Ticket'].str.split().str[0].apply(
        lambda x: 'U0' if x.isnumeric() else x)

    #   Cabin
    combined_train_test['Cabin'].fillna('U0', inplace=True)
    combined_train_test['Cabin'] = combined_train_test['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)

    #   factorize & dummies
    factorize_items = ['Embarked','Sex','Title','Fare_bin','Pclass','Ticket_Letter','Family_Size_Category']
    combined_train_test = factorize_process(combined_train_test,factorize_items)

    dummies_items = ['Embarked','Sex','Title','Fare_bin','Family_Size_Category']
    combined_train_test = dummies_process(combined_train_test,dummies_items)

    '''
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
    '''

    #    Age
    #dummies化所有变量使得Age验证分数上升1%,但最终解决下降了1% 此处临时保留原有Age预测时的数据集，后续优化Age评估器
    #missing_age_df = combined_train_test.drop(['PassengerId','Name', 'Parch', 'SibSp','Ticket'], axis=1)
    missing_age_df = pd.DataFrame(combined_train_test[[
        'Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category', 'Fare', 'Fare_bin',
        'Pclass']])
    missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
    missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]

    def fill_age(X_data,y_data,X_pre):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=0)
        pipe = Pipeline([('preprocessing',StandardScaler()),('classifier',RandomForestRegressor(random_state=42))])
        param_grid = {'classifier__n_estimators':[500],'classifier__max_depth':[5],'classifier__random_state':[0],
                      'classifier__min_samples_leaf':[3],'classifier__n_jobs':[-1]}
        grid = GridSearchCV(pipe,param_grid,cv=5,verbose =2)
        grid.fit(X_train,y_train)
        print('Age feature Best RF Params:' + str(grid.best_params_))
        print('Age feature Best RF Score:' + str(grid.best_score_))
        print('Score: {}'.format(grid.score(X_test, y_test)))
        return grid.predict(X_pre)

    combined_train_test.loc[(combined_train_test['Age'].isnull()), 'Age'] = fill_age(
        missing_age_train.drop(['Age'], axis=1),missing_age_train['Age'],missing_age_test.drop(['Age'], axis=1))

    # 输入模型前的一些处理
    #    1. 弃掉无用特征
    combined_data_backup = combined_train_test.copy()
    combined_train_test.drop(['PassengerId', 'Embarked', 'Sex', 'Name', 'Title', 'Fare_bin',
                              'Parch', 'SibSp', 'Family_Size_Category', 'Ticket'], axis=1, inplace=True)

    #    2. 将训练数据和测试数据分开：
    titanic_train_data_X = combined_train_test[:891].drop(['Survived'], axis=1)
    titanic_train_data_Y = combined_train_test[:891]['Survived']
    titanic_test_data_X = combined_train_test[891:].drop(['Survived'], axis=1)

    # 模型融合及测试
    #    (1) 利用不同的模型来对特征进行筛选，选出较为重要的特征：
    #    (2) 依据我们筛选出的特征构建训练集和测试集 此处不采用 Holdout 而使用分离的KFlod原因:
    #           1.Holdout没有KFlod验证充分
    #           2.分离的KFlod可以对各子测试集/验证集进行评估，而cross_val_score 不够直观
    X_train,X_test,y_train,y_test = train_test_split(titanic_train_data_X,titanic_train_data_Y,random_state=0)

    #管道搜索模型参数
    pipe = Pipeline([('preprocessing',StandardScaler()),('classifier',SVC())])
    grid = GridSearchCV(pipe, svc_model1, cv = 5, n_jobs = -1,verbose = 1,scoring = "roc_auc")
    grid.fit(X_train,y_train)

    print('best score :{}'.format(grid.best_score_))
    print('score :{}'.format(grid.score(X_test,y_test)))
    print('confusion_natrix :{}'.format(confusion_matrix(y_test,grid.predict(X_test))))
    print(classification_report(y_test,grid.predict(X_test),target_names=['not survived','survived']))
    train_sizes, train_scores,test_scores = learning_curve(grid.best_params_['classifier'],
                                                           grid.best_params_['preprocessing'].transform(titanic_train_data_X),
                                                           titanic_train_data_Y,
                                                           cv = 5, train_sizes = np.linspace(0.05,1,20))
    train_mean = np.mean(train_scores, 1)
    train_std = np.std(train_scores, 1)
    test_mean = np.mean(test_scores, 1)
    test_std = np.std(test_scores, 1)
    diff = train_mean - test_mean

    print('train_scores mean: \n {}\n std :\n{}'.format(train_mean, train_std))
    print('test_scores mean: \n {}\n std :\n{}'.format(test_mean, test_std))
    print('train - test :\n{}'.format(diff))

    if PLOT_LEARN_CURVE:
        plt.plot(train_sizes, train_mean, color = 'blue', label = 'training score')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = 0.15, color = 'blue')
        plt.plot(train_sizes, test_mean, color = 'red', label = 'validation score')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha = 0.15, color = 'red')
        plt.plot(train_sizes, diff, color = 'black', label = 'train - val')
        plt.ylim(0,1)
        plt.legend(loc='best')
        plt.title('dummies')
        plt.show()

    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId,
                                       'Survived': grid.predict(titanic_test_data_X)})
    StackingSubmission.to_csv(r'result\rst {} - {}.csv'.format(time.strftime("%H.%M.%S"),time.strftime("%d.%m.%Y"))
                              ,index=False, sep=',')

    select = RFE(RandomForestClassifier(n_estimators=300,max_depth=3,max_features='log2',min_samples_leaf=3))
    select.fit(X_train,y_train)
    X_train_rfe = select.transform(X_train)
    X_test_rfe = select.transform(X_test)
    grid.fit(X_train_rfe,y_train)

    print('best score :{}'.format(grid.best_score_))
    print('score :{}'.format(grid.score(X_test_rfe,y_test)))
    print('confusion_natrix :{}'.format(confusion_matrix(y_test,grid.predict(X_test_rfe))))
    print(classification_report(y_test,grid.predict(X_test_rfe),target_names=['not survived','survived']))
    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': grid.predict(select.transform(titanic_test_data_X))})
    StackingSubmission.to_csv(r'result\rst_RFE {} - {}.csv'.format(time.strftime("%H.%M.%S"),time.strftime("%d.%m.%Y"))
                              ,index=False, sep=',')