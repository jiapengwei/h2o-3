from __future__ import print_function
import sys, os
sys.path.insert(1, os.path.join("..","..",".."))
import h2o
import h2o.exceptions
from tests import pyunit_utils
from h2o.estimators import H2OGeneralizedLinearEstimator

# During normal GLM model building, the coefficient length can shrink when coefficients/gram matrix has zero 
# rows/columns.  Since betaCnd is allocated at the beginning of iteration loop and the coefficient length change
# happened within the iteration loop, there can be a discrepancy in the coefficient lengths.  Normally, this is not a 
# problem because the action of betaCnd = ADMM_solve() or other solvers.  But, in this case, that call is skipped.
# Hence, you will get betaCnd of one length and _state.beta() of another length.  My fix is to make sure when there
# is a length difference, I will extract the correct coefficients from betaCnd such that it will be of the same length
# as _state.beta().
#
# Test provided by Seb.
def test_GLM_throws_ArrayOutOfBoundException():
    fr = h2o.import_file("s3://sebp/datasets/christine.arff")

    glm = H2OGeneralizedLinearEstimator(family='binomial',
                                    nfolds=5,
                                    lambda_search=True,
                                    alpha=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                                    )
    glm.train(y=0, training_frame=fr)

if __name__ == "__main__":
    pyunit_utils.standalone_test(test_GLM_throws_ArrayOutOfBoundException)
else:
    test_GLM_throws_ArrayOutOfBoundException()
