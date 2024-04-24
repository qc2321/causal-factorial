import rpy2
import rpy2.robjects as robjects
r = robjects.r
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
# summary = robjects.r('''summary(fit2)''')
# print(summary)
print('hi')
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
enc = OneHotEncoder(handle_unknown='ignore')
X = carlson_df.iloc[:,1:-2]
y = carlson_df.iloc[:,0]
print(X)
trans_x = enc.fit_transform(X).toarray()
print(trans_x.shape)
degree = 14
pf = preprocessing.PolynomialFeatures(
    degree=degree, interaction_only=True, include_bias=True,
)
T = pf.fit_transform(trans_x)
print(T.shape)
# train test split
T_train, T_test, y_train, y_test = train_test_split(T, y, test_size=0.3, random_state=42)
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
alpha_cv = [0.001, 0.01, 0.1, 0.5]
model = LogisticRegression(penalty='l1',C = .09, solver = 'liblinear')
model.fit(T_train, y_train)

y_pred = model.predict(T_test)
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
# print(model.coef_)
import numpy as np
nonzero_indices = np.nonzero(model.coef_)

# Retrieve the non-zero values using the indices
nonzero_values = model.coef_[nonzero_indices]
# print(enc.get_feature_names_out())
feature_dict = {}
for item in zip(enc.get_feature_names_out(), pf.get_feature_names_out()[1:15]):
    feature_dict[item[1]] = item[0]
print(feature_dict)

for index, value in zip(nonzero_indices[1], nonzero_values):
    features = ""
    for feature in pf.get_feature_names_out()[index].split():
        features += feature_dict[feature] + " "
    print(f"Features: {features}, Value: {value}")
    