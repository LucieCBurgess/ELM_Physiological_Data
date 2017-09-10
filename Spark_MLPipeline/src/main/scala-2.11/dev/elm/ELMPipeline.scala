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

    val fileName: String = "mHealth_subject1.txt"

    /** Load training and test data and cache it */
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

    /**
      * Combine columns which we think will predict labels into a single feature vector
      * NB. Pipeline.fit() command is not picking up the VectorAssembler properly so we have to transform the data outside the pipeline
      */
    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z")
    //FIXME not currently used
    val allowedInputCols: Array[String] = ScalaReflection.schemaFor[MHealthUser].dataType match {
      case s: StructType => s.fieldNames.array
      case _ => Array[String]()
    }

    val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    //pipelineStages += featureAssembler //KEEP THIS OUT - pipelineStages not picking up transform method correctly??

    /** Add the features column, "features", to the input data frame using VectorAssembler */
    val dataWithFeatures = featureAssembler.transform(data)

    /** Create the classifier, set parameters for training */
    val elm = new ELMClassifier()
      .setFeaturesCol("features") // could be "extracted features" if we need to plug in the feature extraction technique
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
    val Array(trainData, testData) = dataWithFeatures.randomSplit(Array(train, test), seed = 12345) // was data

    /** Fit the pipeline, which includes training the model, on the preparedData */
    val startTime = System.nanoTime()

    println(s"************* Training the model ****************")

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

    //println(s"The schema for the predicted dataset based on the training data is ${predictionsTrain.printSchema}")
    //predictionsTrain.printSchema
    println(s"Printing predictions for the training data")
    predictionsTrain.show(10)

    val startTime3 = System.nanoTime()
    val predictionsTest: DataFrame = elmModel.transform(testData).cache()
    val predictTimeTest = (System.nanoTime() - startTime3) / 1e9
    println(s"Prediction time for the test data: $predictTimeTest seconds")
    //println(s"The schema for the predicted dataset based on the test data is ${predictionsTest.printSchema}")
    //predictionsTest.printSchema
    println(s"Printing predictions for the test data")
    predictionsTest.show(10)

    //FIXME - update to include BinaryClassificationEvaluator, once the ELM is working
  }
}

//    def checkInAllowedInputCols(feature: String): String = {
//      if (!allowedInputCols.contains(feature)) throw new IllegalArgumentException("Feature is not in the schema")
//      else featureCols.addString(feature)
//    }
//
