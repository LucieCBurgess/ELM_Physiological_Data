package dev.elm

import dev.pipeline.{LRTestParams, SparkSessionWrapper}
import dev.data_load.{DataLoad, MHealthUser}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator}
import org.apache.spark.ml.linalg.{DenseVector => SDV}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage, Transformer}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

import scala.collection.mutable

/**
  * Created by lucieburgess on 30/08/2017.
  */
object ELMPipeline extends SparkSessionWrapper {

  //FIXME - Not using ELMMain at this stage - need to get working without complicated Params first

  def main(args: Array[String]) {

    import spark.implicits._

    val fileName: String = "mHealth_subject1.txt"

    //def run(params: DefaultELMParams): Unit = {

    /** Load training and test data and cache it */
    val data = DataLoad.createDataFrame(fileName) match {
      case Some(df) => df
        .filter($"activityLabel" > 0)
        .withColumn("binaryLabel", when($"activityLabel".between(1, 3), 0).otherwise(1))
        .withColumn("uniqueID", monotonically_increasing_id())
      case None => throw new UnsupportedOperationException("Couldn't create DataFrame")
    }

    /** Set up the pipeline stages */
    val pipelineStages = new mutable.ArrayBuffer[PipelineStage]()

    /**
      * Combine columns which we think will predict Activity into a single feature vector
      * In this simple example we will just include a few features as a proof of concept
      * Sets the input columns as the array of features and the output column as a new column, a vector modelFeatures
      * Add the featureAssembler to the pipeline
      */
    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z", "acc_Arm_X", "acc_Arm_Y", "acc_Arm_Z")
    /** Parameter for input cols of the ELM, which is combined into a FeaturesCol vector in the Transformer */
    //FIXME not currently used
    val allowedInputCols: Array[String] = ScalaReflection.schemaFor[MHealthUser].dataType match {
      case s: StructType => s.fieldNames
      case _ => Array[String]()
    }


    val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    pipelineStages += featureAssembler //KEEP THIS OUT - pipelineStages not picking up transform method correctly??

    /** Add the features column, "features", to the input data frame as the pipelineStages is not picking this up correctly */
    //val preparedData = featureAssembler.transform(data)

    //val featuresDF = preparedData.toDF("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z", "acc_Arm_X", "acc_Arm_Y", "acc_Arm_Z")

    /** Create the classifier, set parameters for training */
    val elm = new ELMClassifier()
      .setFeaturesCol("features")
      .setLabelCol("binaryLabel")
      .setHiddenNodes(10)
      .setActivationFunc("sigmoid")
      .setFracTest(0.5)

    pipelineStages += elm
    println("ELM parameters:\n" + elm.explainParams() + "\n")

    /** Set the pipeline from the pipeline stages */
    val pipeline: Pipeline = new Pipeline().setStages(pipelineStages.toArray)

    /** UseFracTest to set up the (trainData, testData) tuple and randomly split the preparedData */
    val train: Double = 1-elm.getFracTest
    val test: Double = elm.getFracTest
    val Array(trainData, testData) = data.randomSplit(Array(train, test), seed = 12345) // was preparedData

    /** Fit the pipeline, which includes training the model, on the preparedData */
    val startTime = System.nanoTime()
    val pipelineModel: PipelineModel = pipeline.fit(trainData)
    val elmModel = pipelineModel.stages.last.asInstanceOf[ELMModel]

    println(s"************** Printing the featuresCol ************** + ${elmModel.getFeaturesCol}")


    println("*************** Printing the schema of the training data within ELMPipeline ******************")
    trainData.printSchema

    println("*************** Printing the schema of the test data within ELMPipeline *******************")
    testData.printSchema


    val trainingTime = (System.nanoTime() - startTime) / 1e9
    println(s"Training time: $trainingTime seconds")

    /** Evaluate the model on the training and test data */
    val predictionsTrain = elmModel.transform(trainData).cache()
    println(s"The schema for the predicted dataset based on the training data is ${predictionsTrain.printSchema}")
    predictionsTrain.printSchema
    predictionsTrain.show(10)

    val predictionsTest: DataFrame = elmModel.transform(testData).cache()
    println(s"The schema for the predicted dataset based on the test data is ${predictionsTest.printSchema}")
    predictionsTest.printSchema
    predictionsTest.show(10)

    //FIXME - update to include BinaryClassificationEvaluator, once the ELM is working
  }
}
