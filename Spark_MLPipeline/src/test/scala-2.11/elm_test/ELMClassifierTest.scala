package elm_test

import dev.data_load.DataLoadOption
import dev.elm.{ELMClassifier, ELMParams}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
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
    val extraParam: Param[String] = new Param[String](testELM, "anotherParam", s"another Param for testing purposes")
    val params2 = params.put(extraParam, extraParam.name)

    val testELM2 = testELM.copy(params2)

    println(testELM2.extractParamMap().toSeq.toString())

    assert(testELM2.isInstanceOf[ELMClassifier])
    //assert(testELM2.hasParam("anotherParam"))
  }
}
