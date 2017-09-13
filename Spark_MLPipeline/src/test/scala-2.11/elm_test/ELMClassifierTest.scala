package elm_test

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import dev.data_load.DataLoadOption
import dev.elm.{ELMClassifier, ELMModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{monotonically_increasing_id, when}
import org.scalatest.FunSuite

/**
  * Created by lucieburgess on 12/09/2017.
  * Tests to check logic and methods for ELMClassifier class.
  */
class ELMClassifierTest extends FunSuite {

  val spark: SparkSession = {
    SparkSession.builder().master("local[4]").appName("ELMPipelineTest").getOrCreate()
  }

  import spark.implicits._

  val smallFile: String = "smalltest.txt" //smallFile has 22 rows of data with 3 features

  /** Load training and test data and cache it */
  val smallData = DataLoadOption.createDataFrame(smallFile) match {
    case Some(df) => df
      .filter($"activityLabel" > 0)
      .withColumn("binaryLabel", when($"activityLabel".between(1, 3), 0).otherwise(1))
      .withColumn("uniqueID", monotonically_increasing_id())
    case None => throw new UnsupportedOperationException("Couldn't create DataFrame")
  }

  val featureCols: Array[String] = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z")
  val numFeatures: Int = featureCols.length
  println(s"The number of features calculated from the array featureCols is $numFeatures")
  val featureAssembler: VectorAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
  val dataWithFeatures: DataFrame = featureAssembler.transform(smallData)

  test("[01 Calling new ELMClassifier() creates a new ELMClassifier object with correct parameters") {

    val testELM = new ELMClassifier()
      .setFeaturesCol("features")
      .setLabelCol("binaryLabel")
      .setHiddenNodes(10)
      .setActivationFunc("sigmoid")
      .setFracTest(0.5)

    assert(testELM.getFeaturesCol.equals("features"))
    assert(testELM.getLabelCol.equals("binaryLabel"))
    assert(testELM.getHiddenNodes == 10)
    assert(testELM.getActivationFunc.equals("sigmoid"))
    assert(testELM.getFracTest == 0.5)
    assert(testELM.isInstanceOf[ELMClassifier])
  }

  test("[02 Calling new ELMClassifier returns correct UID") {

    val testELM = new ELMClassifier("ELM_with_UID")
    assertResult("ELM_with_UID") {testELM.uid}
  }

  test("[03 Calling copy copies additional parameters in addition to the default") {

    val testELM = new ELMClassifier()
      .setFeaturesCol("features")
      .setLabelCol("binaryLabel")
      .setHiddenNodes(10)
      .setActivationFunc("sigmoid")
      .setFracTest(0.5)

    val params: ParamMap = testELM.extractParamMap()
    println(params.toSeq.toString())

    // Extra param is not defined in ELMParams so copying it does not work
    val extraParam: Param[String] = new Param[String](testELM, "anotherParam", s"another Param for testing purposes")
    val params2: ParamMap = new ParamMap().put(extraParam, extraParam.name)
    println(params2.toSeq.toString())

    // Copying the embedded params
    val testELM2 = testELM.copy(params)
    println(testELM2.extractParamMap().toSeq.toString())

    assert(testELM2.isInstanceOf[ELMClassifier])
    assert(testELM2.hasParam("featuresCol"))
    assert(testELM2.hasParam("labelCol"))
    assert(testELM2.hasParam("hiddenNodes"))
    assert(testELM2.hasParam("activationFunc"))
    assert(testELM2.hasParam("fracTest"))
    assert(testELM2.hasParam("predictionCol"))
    assert(testELM2.hasParam("rawPredictionCol"))
  }

  test("[04] Calling train with parameters and calling ELMAlgo results in ELMModel with correct parameters") {

    val testELM = new ELMClassifier()
      .setFeaturesCol("features")
      .setLabelCol("binaryLabel")
      .setHiddenNodes(10)
      .setActivationFunc("sigmoid")
      .setFracTest(0.5)

    val testELMModel: ELMModel = testELM.fit(dataWithFeatures)

    assert(testELMModel.isInstanceOf[ELMModel])
    assert(testELMModel.uid.equals(testELM.uid))
    assert(testELMModel.getFracTest == 0.5)
    assert(testELMModel.getHiddenNodes == 10)
    assert(testELMModel.getActivationFunc == "sigmoid")
    assert(testELMModel.getFeaturesCol == "features")
    assert(testELMModel.getLabelCol == "binaryLabel")
    assert(testELMModel.getPredictionCol == "prediction")
    assert(testELMModel.getRawPredictionCol == "rawPrediction")
    println(s"testELMModel.modelNumFeatures is ${testELMModel.modelNumFeatures}")
    println(s"testELMModel.numFeatures is ${testELMModel.numFeatures}")
    assert(testELMModel.modelWeights.isInstanceOf[BDM[Double]])
    assert(testELMModel.modelBias.isInstanceOf[BDV[Double]])
    assert(testELMModel.modelBeta.isInstanceOf[BDV[Double]])

  }

}
