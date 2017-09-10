package dev.logreg

import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util.Identifiable

/**
  * Created by lucieburgess on 25/08/2017.
  * Parameters class for Logistic Regression model specifying the default values
  */
case class LRParams(regParam: Double = 0.0,
                    elasticNetParam: Double = 0.0,
                    maxIter: Int = 100,
                    fitIntercept: Boolean = true,
                    tol: Double = 1E-6,
                    fracTest: Double = 0.5) //extends AbstractParams[LRParams]


