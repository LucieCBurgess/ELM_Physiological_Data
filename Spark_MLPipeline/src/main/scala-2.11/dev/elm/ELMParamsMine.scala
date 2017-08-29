package dev.elm

import dev.data_load.MHealthUser
import org.apache.spark.ml.param._
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.{StructField,StructType}
import org.apache.spark.mllib.linalg.VectorUDT

//FIXME need to understand better how this interacts with the Scopt params version

/**
  * Created by lucieburgess on 27/08/2017.
  * See https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/ml/DeveloperApiExample.scala
  *
  * NOTE: The usual way to add a parameter to a model or algorithm is to include:
  *   - val myParamName: ParamType
  *   - def getMyParamName
  *   - def setMyParamName
  * Here, we have a trait to be mixed in with the Estimator and Model (ELMEstimator and ELMEstimatorModel) respectively.
  * We place the setters e.g (setInputCol) method in the ELMEstimator class since the inputCol parameter is only used during training
  * (not in the Model).
  */
trait ELMParamsMine extends Params {

  /** Defines and sets the activation function parameter and uses the ParamValidators factory methods class to check configuration */
  val activationFuncs: Array[String] = Array("Sigmoid","RBF", "Tanh","Step")
  val activationFunc: Param[String] =
    new Param(this,"activationFunc", s"The activation function which sets modifies the hidden layer output, " +
      s"available activation functions: ${activationFuncs}.toString", ParamValidators.inArray(activationFuncs))
    def getActivationFunc: String = $(activationFunc)

  /** Parameter for the number of hidden nodes in the ELM */
    //FIXME - have no idea if 200 is a reasonable number of hidden nodes, need to check with cross validation
  val hiddenNodes: IntParam =
    new IntParam(this, "hiddenNodes", "number of hidden nodes in the ELM", ParamValidators.inRange(10.0,200.0))
    def getHiddenNodes: Int = $(hiddenNodes)

  /** Parameter for input cols of the ELM, which is combined into a FeaturesCol vector in the Transformer */
  val allowedInputCols: Array[String] = ScalaReflection.schemaFor[MHealthUser].dataType match {
    case s: StructType => s.fieldNames
    case _ => Array[String]()
  }

  val inputCol: Param[String] =
    new Param(this, "inputCol", s"The input column made up of column features", ParamValidators.inArray(allowedInputCols))
    def getInputCol: String = $(inputCol)

  /** Parameter for the output col of the ELM, which can be set by the user */
  val outputCol: Param[String] =
    new Param(this, "inputCol", s"The input column made up of column features")
  def getOutputCol: String = $(outputCol)

  /** Parameter to set the fraction of the dataset to be held out for testing. Can be in range 10% - 50%
    * At least 50% must be used to train the model
    */
  val fracTest: DoubleParam =
    new DoubleParam(this, "fracTest", "Fraction of data to be held out for testing", ParamValidators.inRange(0.10, 0.50))
    //def setFracTest(value: Double): this.type = set(fracTest, value)
    def getFracTest: Double = $(fracTest)

  /** Validate and transform the input schema, which can then be used by ELMEstimator and ELMEstimatorModel
    * https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/feature/MinMaxScaler.scala
    */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    ELMSchemaUtils.checkColumnType(schema, $(inputCol), new VectorUDT)
    require(!schema.fieldNames.contains($(outputCol)), s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), new VectorUDT, false)
    StructType(outputFields)
  }

}
