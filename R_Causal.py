import rpy2
import rpy2.robjects as robjects
r = robjects.r
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

base = importr('base')
utils = importr('utils')
r('install.packages("FindIt", repos="https://CRAN.R-project.org/")')
findit = importr('FindIt')

pi = robjects.r['pi']
print(pi[0])
rsum = robjects.r['sum']
print(rsum(robjects.IntVector([1,2,3]))[0])


robjects.r('data("Carlson", package = "FindIt")')
carlson = robjects.r['Carlson']
carlson_df = pandas2ri.rpy2py(carlson)
print(carlson_df)