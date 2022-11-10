target = 'TARGET'

model_features = training_data.columns[~training_data.columns.isin(nonmodel_cols+[target])].tolist()
model_features_numeric = [x for x in model_features if training_data[x].dtype in ['int64' ,'float']]
model_features_categorical = list(set(model_features) - set(model_features_numeric))

X = training_data[model_features]
y = training_data[[target]]
holdout_size = .1
test_size = .2
seed = 88
knn=10

X_train, X_holdout, y_train, y_holdout = train_test_split(X,y,test_size=holdout_size, random_state=seed)
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=test_size/(1-holdout_size), random_state=seed)

print("Holdout Split Percentage: {}".format(holdout_size))
print("Test Split Percentage: {}".format(test_size))
print("Train Percentage: {}".format(1-test_size-holdout_size))
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore',sparse=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, model_features_numeric),
        ('cat', categorical_transformer, model_features_categorical)])

preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])


lgbm = lgb.LGBMClassifier(random_state=42, n_jobs=-1, objective='binary', n_estimators=150)


pipeline = Pipeline(steps = [
            ('preprocessor', preprocessor),
            ('clf',lgbm)
           ])

pipeline_temp = Pipeline(pipeline.steps[:-1])  
X_train_trans = pipeline_temp.fit_transform(X_train)
X_test_trans = pipeline_temp.transform(X_test)

model_features_categorical_ohe = pipeline_temp['preprocessor'].transformers_[1][1]['onehot'].get_feature_names(model_features_categorical)
model_columns = model_features_numeric + model_features_categorical_ohe.tolist()

X_train_trans = pd.DataFrame(X_train_trans, columns = model_columns,index=X_train.index)
X_test_trans = pd.DataFrame(X_test_trans, columns = model_columns,index=X_test.index)

eval_set = [(X_train_trans, y_train), (X_test_trans, y_test)]

param_space =  {
    'clf__max_depth': [2, 3,4,5],
    "clf__n_estimators": [20,25,30,50,75],
    'clf__num_leaves': [7, 8, 9],
}

evals_result = {}

fit_params = {
    'clf__early_stopping_rounds':30,
    'clf__eval_metric':'accuracy',
    'clf__eval_set':eval_set,
    'clf__evals_result':evals_result
}

opt = BayesSearchCV(
    pipeline,
    param_space,
    n_iter=5,
    cv=5,
    scoring='neg_log_loss',
    verbose=3, 
    n_jobs=-1,
    random_state=88,
    fit_params=fit_params
)

opt.fit(X_train, y_train.values.ravel())
print(
opt
.
best_estimator_)






search_results = opt.optimizer_results_[0]['func_vals']
search_results = pd.DataFrame(results,columns=['search_results'])
search_results = search_results.reset_index()
search_results

print(pipeline.feature_names_in_)

lgbm_feature_importance = pd.DataFrame()
lgbm_feature_importance['feature'] = model_columns
values = opt.best_estimator_.steps[-1][1].feature_importances_.tolist()

lgbm_feature_importance['importance'] = values
lgbm_feature_importance = lgbm_feature_importance.sort_values('importance',ascending=False)

display(lgbm_feature_importance.head(5))
sns.barplot(data=lgbm_feature_importance.head(10), y='feature',x='importance',palette='mako')

opt.best_params_
opt.best_score_
metrics = ['log_loss','auc','r2']
data = ['train','test','holdout']
model_results = pd.DataFrame(columns=data,index=metrics)
roc_table = pd.DataFrame(columns=['models', 'fpr','tpr','auc'])


for x in data:
    df = X_train if x == 'train' else X_test if x=='test' else X_holdout
    y = y_train if x == 'train' else y_test if x=='test' else y_holdout

    pred = pd.DataFrame(opt.predict_proba(df))[1]
    model_results.at['auc',x] = roc_auc_score(y,pred)
    model_results.at['log_loss',x] = log_loss(y,pred)
    model_results.at['r2',x] = r2_score(y,pred)

    
pred = pd.DataFrame(opt.predict_proba(X_test))[1]
fpr, tpr, _ = roc_curve(y_test,  pred)
auc = roc_auc_score(y_test, pred)

roc_table = roc_table.append({'models':'lightgbm',
                                    'fpr':fpr, 
                                    'tpr':tpr, 
                                    'auc':auc}, ignore_index=True)

display(model_results)


fig = plt.figure(figsize=(8,6))
plt.plot(roc_table.loc[0]['fpr'], 
         roc_table.loc[0]['tpr'], 
         label="{}, AUC={:.3f}".format(i, roc_table.loc[0]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve by Model', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()






explainer = shap.TreeExplainer(opt.best_estimator_['clf'])
shap_values = explainer(X_train_trans)
shap.plots.bar(shap_values[:,:,1])
shap.summary_plot(shap_values[:,:,1], X_train_trans)

## SAVE the pipeline "opt" example

joblib.dump(opt, f"{training_run_model_path}/model.pkl")