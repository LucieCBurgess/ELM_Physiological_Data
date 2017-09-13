package dev.elm

import dev.logreg.LRParams
import dev.data_load.{DataLoadOption, MHealthUser}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator}
import org.apache.spark.ml.linalg.{DenseVector => SDV}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage, Transformer}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

import scala.collection.mutable

/**
  * Created by lucieburgess on 30/08/2017.
  */
object ELMPipeline {

  //FIXME - Not using ELMMain at this stage - need to get working without complicated Params first

  lazy val spark: SparkSession = {
    SparkSession.builder().master("local[4]").appName("ELMPipeline").getOrCreate()
  }

  def main(args: Array[String]) {

    import spark.implicits._

    val fileName: String = "smalltest.txt"

    /** Load training and test data and cache it
      * A single method only is used here for one input dataset
      * //FIXME - add the multiple dataload option to see if the method works on more than one file on the command line
      */
    val data = DataLoadOption.createDataFrame(fileName) match {
      case Some(df) => df
        .filter($"activityLabel" > 0.0)
        .withColumn("binaryLabel", when($"activityLabel".between(1.0, 3.0), 0.0).otherwise(1.0))
        .withColumn("uniqueID", monotonically_increasing_id())
      case None => throw new UnsupportedOperationException("Couldn't create DataFrame")
    }

    val Nsamples: Int = data.count().toInt
    println(s"The number of training samples is $Nsamples")

    /** Set up the pipeline stages */
    val pipelineStages = new mutable.ArrayBuffer[PipelineStage]()

    /** Combine columns which we think will predict labels into a single feature vector */
    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z")
    checkFeatureColsInSchema(featureCols)

    /**
      * Add the features to the DataFrame using VectorAssembler.
      * Pipeline.fit() command is not picking up the VectorAssembler properly so we have to transform the data outside the pipeline
      * This has to be done outside pipelineStages due to a bug in the API
      */
    val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val dataWithFeatures = featureAssembler.transform(data)

    /** Create the classifier, set parameters for training */
    val elm = new ELMClassifier()
      .setFeaturesCol("features")
      .setLabelCol("binaryLabel")
      .setHiddenNodes(10)
      .setActivationFunc("sigmoid") //was sigmoid
      .setFracTest(0.5)

    pipelineStages += elm
    println("ELM parameters:\n" + elm.explainParams() + "\n")

    /** Set the pipeline from the pipeline stages */
    val pipeline: Pipeline = new Pipeline().setStages(pipelineStages.toArray)

    /** UseFracTest to set up the (trainData, testData) tuple and randomly split the preparedData */
    val train: Double = 1 - elm.getFracTest
    val test: Double = elm.getFracTest
    val Array(trainData, testData) = dataWithFeatures.randomSplit(Array(train, test), seed = 12345)

    /** Fit the pipeline, which includes training the model, on the preparedData */
    val startTime = System.nanoTime()

    val pipelineModel: PipelineModel = pipeline.fit(trainData)
    val elmModel = pipelineModel.stages.last.asInstanceOf[ELMModel]

    println(s"The schema of the training data is ${trainData.printSchema}")
    trainData.printSchema

    println(s"The schema of the test data is ${testData.printSchema}")
    testData.printSchema

    val trainingTime = (System.nanoTime() - startTime) / 1e9
    println(s"Training time: $trainingTime seconds")

    /** Evaluate the model on the training and test data */
    val startTime2 = System.nanoTime()
    val predictionsTrain = elmModel.transform(trainData).cache()
    val predictTimeTrain = (System.nanoTime() - startTime2) / 1e9
    println(s"Prediction time for the training data: $predictTimeTrain seconds")
    println(s"Printing predictions for the training data")
    predictionsTrain.show(10)

    val startTime3 = System.nanoTime()
    val predictionsTest: DataFrame = elmModel.transform(testData).cache()
    val predictTimeTest = (System.nanoTime() - startTime3) / 1e9
    println(s"Prediction time for the test data: $predictTimeTest seconds")
    println(s"Printing predictions for the test data")
    predictionsTest.show(10)

    //FIXME - update to include BinaryClassificationEvaluator, once the ELM is working
  }

  /** Helper method to check the selected feature columns are in the schema
    * @throws IllegalArgumentException if the selected feature columns are not in the schema
    */
  def checkFeatureColsInSchema(featureCols: Array[String]): Array[String] = {
    val allowedInputCols: Array[String] = ScalaReflection.schemaFor[MHealthUser].dataType match {
      case s: StructType => s.fieldNames.array
      case _ => Array[String]()
    }
    val result = featureCols.map(c => allowedInputCols.contains(c))
    if (result.contains(false)) throw new IllegalArgumentException("Feature cols not in schema")
    else featureCols
  }
}