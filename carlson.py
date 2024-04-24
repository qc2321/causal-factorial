import rpy2
import rpy2.robjects as robjects
r = robjects.r
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import statsmodels.api as sm
import pandas as pd
from itertools import combinations

base = importr('base')
utils = importr('utils')
r('install.packages("FindIt", repos="https://CRAN.R-project.org/")')
findit = importr('FindIt')

'''messing around with how rpy2 works'''
pi = robjects.r['pi']
print(pi[0])
rsum = robjects.r['sum']
print(rsum(robjects.IntVector([1,2,3]))[0])


'''using data from example in FindIt'''
robjects.r('data("Carlson", package = "FindIt")')
carlson = robjects.r['Carlson']
carlson_df = pandas2ri.rpy2py(carlson)
print(carlson_df)

'''trying a couple variables to use the 2 way amie'''
new_Record = robjects.r('''
Carlson$newRecordF<- factor(Carlson$newRecordF,ordered=TRUE,
levels=c("YesLC", "YesDis","YesMP",
"noLC","noDis","noMP","noBusi"))''')
promise = robjects.r('''
Carlson$promise <- factor(Carlson$promise,ordered=TRUE,levels=c("jobs","clinic","education"))
''')
model = robjects.r('''

fit2 <- CausalANOVA(formula=won ~ newRecordF + promise,
int2.formula = ~ newRecordF:promise,
data=Carlson, pair.id=Carlson$contestresp,diff=TRUE,
cluster=Carlson$respcodeS, nway=2)
''')


#one hot encode
enc = OneHotEncoder(handle_unknown='ignore')
X = carlson_df.iloc[:,1:-2]
y = carlson_df.iloc[:,0].to_numpy()
trans_x = enc.fit_transform(X).toarray()
print(type(trans_x))
print(type(y))
#expand for fourier
degree = 14
pf = preprocessing.PolynomialFeatures(
    degree=degree, interaction_only=True, include_bias=True,
)
T = pf.fit_transform(trans_x) * 2 - 1


# train test split
T_train, T_test, y_train, y_test = train_test_split(T, y, test_size=0.3, random_state=42)
print(T_train)
print(y_train)
# alpha_cv = [0.001, 0.01, 0.1, 0.5]
#logistic regression
model = LogisticRegression(penalty='l1',C = .06, solver = 'liblinear')
model.fit(T_train, y_train)

y_pred = model.predict(T_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"accuracy {accuracy}")
# print(model.coef_)
import numpy as np
nonzero_indices = np.nonzero(model.coef_)

# Retrieve the non-zero values using the indices
nonzero_values = model.coef_[nonzero_indices]
# print(enc.get_feature_names_out())
# feature_dict = {}
# for item in zip(enc.get_feature_names_out(), pf.get_feature_names_out()[1:15]):
#     feature_dict[item[1]] = item[0]
# print(feature_dict)

# for index, value in zip(nonzero_indices[1], nonzero_values):
#     features = ""
#     for feature in pf.get_feature_names_out()[index].split():
#         features += feature_dict[feature] + " "
#     print(f"Features: {features}, Value: {value}")
    

# linear regression
model2 = Lasso(alpha=0.011)
model2.fit(T_train, y_train)
y_pred = model2.predict(T_test)
# Retrieve the non-zero values using the indices
nonzero_indices = np.nonzero(model2.coef_)
nonzero_values = model2.coef_[nonzero_indices]
# print(enc.get_feature_names_out())
feature_dict = {}
for item in zip(enc.get_feature_names_out(), pf.get_feature_names_out()[1:15]):
    feature_dict[item[1]] = item[0]
# print(feature_dict)

for index, value in zip(nonzero_indices[0], nonzero_values):
    features = ""
    for feature in pf.get_feature_names_out()[index].split():
        features += feature_dict[feature] + " "
    print(f"Features: {features}, Value: {value}")
count = y_test.shape[0]
correct = 0
for pred, act in zip(y_pred,y_test):
    if (pred >=0.5 and act ==1) or (pred <0.5 and act ==0):
        correct+=1
print(correct/count)




#for forward selection
T = trans_x * 2 - 1
T_train, T_test, y_train, y_test = train_test_split(T, y, test_size=0.3, random_state=42)
T_train = pd.DataFrame(T_train)
y_train = pd.DataFrame(y_train)


def forward_selection(X, y, p_value):
    kept_features = []
    total_features = X.columns.tolist()
    print(total_features)
    # print(y)
    # print(X)

    while True:
        remaining_features = list(set(total_features) - set(kept_features))

        new_pval = pd.Series(index=remaining_features)
        
        for new_column in remaining_features:
            # print(X[kept_features + [new_column]])
            test_columns = list(set([new_column]) | set(kept_features))
            model = sm.OLS(y, sm.add_constant(X[test_columns])).fit(disp=0)
            new_pval[new_column] = model.pvalues[new_column]
        
        min_p_value = new_pval.min()
        if min_p_value < p_value:
            kept_features.append(new_pval.idxmin())
        else:
            break
    return kept_features

def interactions(X, y, p_value, features):
    original_features = features
    combo_features = []
    for feature1, feature2 in combinations(original_features,2):
        print(feature1, feature2)
        X[f"{feature1}*{feature2}"] = X[feature1] * X[feature2]
        combo_features.append(f"{feature1}*{feature2}")
    kept_features = []
    # print(y)
    # print(X)
    X = X * 2 - 1
    # print(X)
    while True:
        remaining_features = list(set(combo_features) - set(kept_features))

        new_pval = pd.Series(index=remaining_features)
        
        for new_column in remaining_features:
            # print(X[kept_features + [new_column]])
            test_columns = list(set(original_features) | set([new_column]) | set(kept_features))
            model = sm.OLS(y, sm.add_constant(X[test_columns])).fit(disp=0)
            new_pval[new_column] = model.pvalues[new_column]
        
        min_p_value = new_pval.min()
        if min_p_value < p_value:
            kept_features.append(new_pval.idxmin())
        else:
            break
    return kept_features

kept_features = forward_selection(T_train,y_train,0.06)
# print(forward_selection(T_train, y_train, 0.06))
print(feature_dict)
print(kept_features)
T = trans_x
print(T.shape)
print(y.shape)

T_train, T_test, y_train, y_test = train_test_split(T, y, test_size=0.3, random_state=42)
T_train = pd.DataFrame(T_train)
y_train = pd.DataFrame(y_train)
print(T_train[kept_features])
print(interactions(T_train, y_train, 0.03, kept_features))



