package elm_test

import dev.data_load.DataLoadOption
import dev.elm.{ELMClassifier, ELMModel}
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorSlicer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, monotonically_increasing_id, when}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.scalatest.FunSuite

import scala.collection.mutable

/**
  * Created by lucieburgess on 02/09/2017.
  * Tests for ELM Pipeline.
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
    ??? //Not yet implemented
  }

  test("[03] Can recreate features using Vector slicer") {

    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z", "acc_Arm_X", "acc_Arm_Y", "acc_Arm_Z")
    val featureColsIndex = featureCols.map(c => s"${c}_index")

    val indexers = featureCols.map(
      c => new StringIndexer().setInputCol(c).setOutputCol(s"${c}_index")
    )

    val assembler = new VectorAssembler().setInputCols(featureColsIndex).setOutputCol("features")
    val slicer = new VectorSlicer().setInputCol("features").setOutputCol("double_features").setNames(featureColsIndex.init)
    val transformedData = new Pipeline().setStages(indexers :+ assembler :+ slicer)
      .fit(smallData)
      .transform(smallData)

    // assert(slicer.getOutputCol.sameElements(assembler.getOutputCol)) Doesn't work
    transformedData.show()
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
      .setFracTest(0.6)

    val train: Double = 1 - elmTest.getFracTest
    val test: Double = elmTest.getFracTest
    val Array(trainData, testData) = dataWithFeatures.randomSplit(Array(train, test), seed = 12345)

    assert(trainData.count()+testData.count() === bigData.count())
  }

  test("[06] pipeline calls behave as expected and return data structures of the correct size") {

    val pipelineStages = new mutable.ArrayBuffer[PipelineStage]()
    assert(pipelineStages.isEmpty)

    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z", "acc_Arm_X", "acc_Arm_Y", "acc_Arm_Z")
    val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val dataWithFeatures = featureAssembler.transform(bigData)

    val elmTest = new ELMClassifier()
      .setFeaturesCol("features")
      .setLabelCol("binaryLabel")
      .setHiddenNodes(10)
      .setActivationFunc("sigmoid")
      .setFracTest(0.6)

    pipelineStages += elmTest
    assert(pipelineStages.size === 1)

    val pipelineTest: Pipeline = new Pipeline().setStages(pipelineStages.toArray)
    assert(pipelineTest.getStages.length === 1)
    assert(pipelineTest.getStages.head.isInstanceOf[ELMClassifier])

    val pipelineTestModel: PipelineModel = pipelineTest.fit(smallData)
    assert(pipelineTestModel.stages.last.isInstanceOf[ELMModel])

    val elmTestModel = pipelineTestModel.stages.last.asInstanceOf[ELMModel]
    assert(elmTestModel.isInstanceOf[ELMModel])

    val testPredictions = elmTestModel.transform(dataWithFeatures)
    assert(testPredictions.count() == dataWithFeatures.count())
  }

}

//val features: RDD[Vector] = dataWithFeatures.select("features").rdd.map(_.getAs[Vector]("features"))
// sbt "testOnly *ELMPipelineTest"