"""
Run R scripts in python
"""

import pandas as pd
import rpy2.robjects as robj
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
pandas2ri.activate()


# Run simple model and load result into python
robj.r("""
    library(data.table)
    dt = data.table(
        x = c(1,2,3,4),
        y = c(5,3,2,0)
    )
    model = lm(y ~ x, data=dt)
    pred = predict(model)
""")
pred = robj.r['pred']


# Put pandas df through an R function
df = pd.DataFrame({'x': [1,2,3,4], 'y': [5,3,2,0]})
with localconverter(robj.default_converter + pandas2ri.converter):
  r_df = robj.conversion.py2rpy(df)
robj.r("""
    run_reg <- function(data) {
       model <- lm(y ~ x, data=data)
       return(predict(model))
    }
    """
)
r_fit = robj.r.run_reg(r_df)
r_fit