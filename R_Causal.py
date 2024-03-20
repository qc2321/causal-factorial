import rpy2
import rpy2.robjects as robjects
r = robjects.r
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

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
# print(carlson_df)

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
summary = robjects.r('''summary(fit2)''')
print(summary)