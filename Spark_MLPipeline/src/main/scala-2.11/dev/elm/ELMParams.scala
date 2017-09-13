package dev.elm

import dev.data_load.MHealthUser
import org.apache.spark.ml.param._
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.{StructField,StructType}
import org.apache.spark.mllib.linalg.VectorUDT

/**
  * Created by lucieburgess on 27/08/2017.
  * See https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/ml/DeveloperApiExample.scala
  *
  * This class contains EXTRA parameters that are not already set by Classifier.scala
  * Classifier sets four parameters which can be get or set but their implementation cannot be overriden as they are final
  * These parameters are: featuresCol, labelCol, predictionCol and rawPredictionCol.
  *
  * NOTE: The usual way to add a parameter to a model or algorithm is to include:
  *   - val myParamName: ParamType
  *   - def getMyParamName
  *   - def setMyParamName
  * Here, we have a trait to be mixed in with the Estimator and Model (ELMClassifier and ELMModel) respectively.
  * We can place the setters e.g (setActivationFunc) method in the ELMClassifier class since the inputCol parameter is only used during training
  * (not in the Model).
  *
  * There is no need to validate and transform the schema as this is already done by ClassifierParams and PredictorParams
  * The method can be overwritten if necessary - see Classifier.scala
  *
  */
trait ELMParams extends Params {

  /** Defines and sets the activation function parameter and uses the ParamValidators factory methods class to check configuration */
  val activationFuncs: Array[String] = Array("sigmoid","tanh","sin","step") // NB all lower case
  val activationFunc: Param[String] =
    new Param[String](this,"activationFunc", s"The activation function which sets modifies the hidden layer output, " +
      s"available activation functions: sigmoid, tanh, sin, step", ParamValidators.inArray(activationFuncs))
    def getActivationFunc: String = $(activationFunc)

  /** Parameter for the number of hidden nodes in the ELM */
  val hiddenNodes: IntParam =
    new IntParam(this, "hiddenNodes", "number of hidden nodes in the ELM", ParamValidators.inRange(10.0,200.0))
    def getHiddenNodes: Int = $(hiddenNodes)

  /**
    * Parameter to set the fraction of the dataset to be held out for testing. Can be in range 10% - 50%
    * At least 50% must be used to train the model
    * @throws IllegalArgumentException if fracTest is <0.10 or >0.50
    */
  val fracTest: DoubleParam =
    new DoubleParam(this, "fracTest", "Fraction of data to be held out for testing", ParamValidators.inRange(0.10, 0.50))
    def getFracTest: Double = $(fracTest)

}
