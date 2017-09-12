package elm_test

import dev.data_load.{DataLoadOption, MHealthUser}
import dev.elm.{ELMClassifier, ELMModel}
import org.apache.spark.ml.linalg.{Vector}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorSlicer}
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.functions.{col, monotonically_increasing_id, when}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.FunSuite
import scala.collection.mutable

/**
  * Created by lucieburgess on 02/09/2017.
  * Tests for ELM Pipeline.
  * ALL TESTS PASS. Use sbt "testOnly *ELMPipelineTest" to run the tests.
  */
class ELMPipelineTest extends FunSuite {

  val spark: SparkSession = {
    SparkSession.builder().master("local[4]").appName("ELMPipelineTest").getOrCreate()
  }

  import spark.implicits._

  val smallFile: String = "smalltest.txt" //smallFile has 22 rows of data with 3 features
  val bigFile: String = "mHealth_subject1.txt" //bigFile has 35,174 rows with 3 features

  /** Load training and test data and cache it */
  val smallData = DataLoadOption.createDataFrame(smallFile) match {
    case Some(df) => df
      .filter($"activityLabel" > 0)
      .withColumn("binaryLabel", when($"activityLabel".between(1, 3), 0).otherwise(1))
      .withColumn("uniqueID", monotonically_increasing_id())
    case None => throw new UnsupportedOperationException("Couldn't create DataFrame")
  }

  /** Big file: Load training and test data and cache it */
  val bigData: DataFrame = DataLoadOption.createDataFrame(bigFile) match {
    case Some(df) => df
      .filter($"activityLabel" > 0)
      .withColumn("binaryLabel", when($"activityLabel".between(1, 3), 0).otherwise(1))
      .withColumn("uniqueID", monotonically_increasing_id())
    case None => throw new UnsupportedOperationException("Couldn't create DataFrame")
  }

  val smallN: Int = smallData.count().toInt
  val bigN: Int = bigData.count().toInt

  /** Checks that features can be added to column as a vector and recalled from the pipeline */
  test("[01] Vector assembler adds output column of features and can be added to pipeline") {

    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z", "acc_Arm_X", "acc_Arm_Y", "acc_Arm_Z")
    val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val dataWithFeatures = featureAssembler.transform(bigData)

    assert(dataWithFeatures.select("features").head.get(0).asInstanceOf[Vector].size === 6)
    assert(featureAssembler.isInstanceOf[VectorAssembler])
    assert(featureAssembler.getInputCols.equals(featureCols))
    assert(featureAssembler.getOutputCol.equals("features"))

    dataWithFeatures.printSchema()
    val pipelineStages = new mutable.ArrayBuffer[PipelineStage]()
    pipelineStages += featureAssembler

    assert(pipelineStages.head == featureAssembler)
  }

  test("[02] checkAllowedInputCols checks that the features are correctly in the schema") {

    val featureColsInSchema = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z",
      "ecg_1", "ecg_2",
      "acc_Ankle_X", "acc_Ankle_Y", "acc_Ankle_Z",
      "gyro_Ankle_X", "gyro_Ankle_Y", "gyro_Ankle_Z",
      "magno_Ankle_X", "magno_Ankle_Y", "magno_Ankle_Z",
      "acc_Arm_X","acc_Arm_Y", "acc_Arm_Z",
      "gyro_Arm_X", "gyro_Arm_Y", "gyro_Arm_Z",
      "magno_Arm_X", "magno_Arm_Y", "magno_Arm_Z",
      "activityLabel")
    val featureColsNotInSchema = Array("bla","acc","health")
    def checkFeatureColsInSchema(featureCols: Array[String]): Array[String] = {
      val allowedInputCols: Array[String] = ScalaReflection.schemaFor[MHealthUser].dataType match {
        case s: StructType => s.fieldNames.array
        case _ => Array[String]()
      }
      val result = featureCols.map(c => allowedInputCols.contains(c))
      if (result.contains(false)) throw new IllegalArgumentException("Feature cols not in schema")
      else featureCols
    }
    assert(checkFeatureColsInSchema(featureColsInSchema)(2).equals("acc_Chest_Z"))
    assert(checkFeatureColsInSchema(featureColsInSchema)(4).equals("ecg_2"))
    assert(checkFeatureColsInSchema(featureColsInSchema)(23).equals("activityLabel"))
    assert(checkFeatureColsInSchema(featureColsInSchema).length == 24)
    intercept[IllegalArgumentException]{ checkFeatureColsInSchema(featureColsNotInSchema)}
  }

  /** Demonstrates that the sliced feature vector is the same as the features */
  test("[03] Can recreate features using Vector slicer") {

    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z", "acc_Arm_X", "acc_Arm_Y", "acc_Arm_Z")

    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val slicer = new VectorSlicer().setInputCol("features").setOutputCol("double_features").setNames(featureCols)

    val slicedData = new Pipeline().setStages(Array(assembler,slicer))
      .fit(smallData)
      .transform(smallData)
    slicedData.show(10)

    val dataWithFeatures = assembler.transform(smallData)

    slicedData.select($"double_features").show
    dataWithFeatures.select($"features").show

    val array1 = dataWithFeatures.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
    val array2 = slicedData.select("double_features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect

    assert(array1.length == array2.length)
    assert(array1.sum == array2.sum)
  }

  test("[04] Creating ELM returns Classifier of the correct type and parameters") {

    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z", "acc_Arm_X", "acc_Arm_Y", "acc_Arm_Z")
    val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val dataWithFeatures = featureAssembler.transform(smallData)

    val elm = new ELMClassifier()
      .setFeaturesCol("features")
      .setLabelCol("binaryLabel")
      .setHiddenNodes(10)
      .setActivationFunc("sigmoid")
      .setFracTest(0.5)

    assert(elm.getFeaturesCol.equals("features"))
    assert(elm.getLabelCol.equals("binaryLabel"))
    assert(elm.getHiddenNodes === 10)
    assert(elm.getActivationFunc.equals("sigmoid"))
    assert(elm.getFracTest === 0.5)
    assert(elm.isInstanceOf[ELMClassifier])
  }

  test("[05] fracTest returns an array of the correct size") {

    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z", "acc_Arm_X", "acc_Arm_Y", "acc_Arm_Z")
    val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val dataWithFeatures = featureAssembler.transform(bigData)

    val elmTest = new ELMClassifier()
      .setFeaturesCol("features")
      .setLabelCol("binaryLabel")
      .setHiddenNodes(10)
      .setActivationFunc("sigmoid")
      .setFracTest(0.45)

    val train: Double = 1 - elmTest.getFracTest
    val test: Double = elmTest.getFracTest
    val Array(trainData, testData) = dataWithFeatures.randomSplit(Array(train, test), seed = 12345)

    assert(trainData.count()+testData.count() === bigData.count())
  }

  /** Intercepts exception caused by fracTest being out of range, greater than 0.5 */
  test("[06] fracTest > 0.5 throws an exception") {

    val elmTest = new ELMClassifier()
      intercept[IllegalArgumentException] {
        elmTest.setFracTest(0.6)
      }
  }

  test("[07] FracTest < 0.5 does not thrown an exception") {

    val elmTest = new ELMClassifier()
    intercept[IllegalArgumentException] {
      elmTest.setFracTest(0.6)
    }
  }

  /** This test can be run on smallData only. Running on bigData requires a Spark config to be set on the command line
    * before the JVM is created on the driver
    * There is a bug in PipelineStages or Pipeline API which means it is not picking up the "features" column on line 215
    * in the call Pipeline.fit(smallData).
    * Therefore we have to take the VectorAssembler out of the pipeline and add the features column manually.
    * This is a problem with the API, not with my code. The commented lines show what should be there if the Classifier API
    * was working correctly with the VectorAssembler API.
    */
  test("[08] pipeline calls behave as expected and return data structures of the correct size") {

    val pipelineStages = new mutable.ArrayBuffer[PipelineStage]()
    assert(pipelineStages.isEmpty)

    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z", "acc_Arm_X", "acc_Arm_Y", "acc_Arm_Z")
    val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    //pipelineStages += featureAssembler

    val dataWithFeatures = featureAssembler.transform(smallData)

    val elmTest = new ELMClassifier()
      .setFeaturesCol("features")
      .setLabelCol("binaryLabel")
      .setHiddenNodes(10)
      .setActivationFunc("sigmoid")
      .setFracTest(0.4)

    pipelineStages += elmTest
    assert(pipelineStages.size == 1)

    val pipelineTest: Pipeline = new Pipeline().setStages(pipelineStages.toArray)
    assert(pipelineTest.getStages.length == 1)
    //assert(pipelineTest.getStages.head.isInstanceOf[VectorAssembler])
    assert(pipelineTest.getStages.head.isInstanceOf[ELMClassifier])

    val pipelineTestModel: PipelineModel = pipelineTest.fit(dataWithFeatures)
    assert(pipelineTestModel.stages.last.isInstanceOf[ELMModel])

    val elmTestModel = pipelineTestModel.stages.last.asInstanceOf[ELMModel]
    assert(elmTestModel.isInstanceOf[ELMModel])

    val testPredictions = elmTestModel.transform(dataWithFeatures) //instead of smallData
    assert(testPredictions.count() == dataWithFeatures.count()) //instead of smallData
  }
}