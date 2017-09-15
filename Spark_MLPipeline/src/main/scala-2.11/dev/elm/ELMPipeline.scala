package dev.elm

import dev.data_load.{DataLoadOption, MHealthUser}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator}
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

  /** Amend singleFileUsed to false if more than a single file is being used
    * Amend "Multiple3" to "Multiple10" if running pipeline across full dataset of 10 users
    */
  val singleFileName: String = "smalltest.txt"
  val singleFileUsed: Boolean = true
  val multipleFolder: String = "Multiple3"

  def run(params: DefaultELMParams): Unit = {

    lazy val spark: SparkSession = {
      SparkSession.builder().master("local[*]").appName(s"ELM pipeline with $params").getOrCreate()
    }

    import spark.implicits._

    println(s"Extreme Learning Machine applied to the mHealth data with parameters: \n$params")

    /** Load training and test data and cache it
      * A single method only is used here for one input dataset
      * //FIXME - add the multiple dataload option to see if the method works on more than one file on the command line
      */
    val data = DataLoadOption.createDataFrame(singleFileName) match {
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
      .setHiddenNodes(params.hiddenNodes)
      .setActivationFunc(params.activationFunc) //was sigmoid
      .setFracTest(params.fracTest)

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

    /** Make predictions and evaluate the model using BinaryClassificationEvaluator */
    println("Evaluating model and calculating train and test AUROC - larger is better")
    evaluateClassificationModel("Train",pipelineModel, trainData) // using elmModel instead of pipelineModel
    evaluateClassificationModel("Test", pipelineModel, testData)

    /** Perform cross-validation on the dataset */
    println("Performing cross validation and computing best parameters")
    performCrossValidation(trainData, testData, params, pipeline, elm)

    spark.stop()
  }

  /** Utilities for methods above *************/

  /** Helper method to check the selected feature columns are in the schema
    *
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

  /**
    * Singleton version of BinaryClassificationEvaluator so we can use the same instance across the whole model
    *  metric must be "areaUnderROC" or "areaUnderPR" according to BinaryClassificationEvaluator API
    */
  private object SingletonEvaluator {
    val elmEvaluator: Evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setLabelCol("binaryLabel")
      .setRawPredictionCol("prediction") //NB. must be "prediction" not rawPrediction otherwise BinaryClassificationEvaluator expects a vector of length 2.

    def getEvaluator: Evaluator = elmEvaluator
  }

  /**
    * Evaluate the given ClassificationModel on data. Print the results
    * @param model  Must fit ClassificationModel abstraction, with Transformers and Estimators
    * @param df  DataFrame with "prediction" and labelColName columns
    */
  private def evaluateClassificationModel(modelName: String, model: Transformer, df: DataFrame): Unit = {

    val startTime = System.nanoTime()
    val predictions = model.transform(df).cache() // gives predictions for both training and test data
    val predictionTime = (System.nanoTime() - startTime) / 1e9
    println(s"Running time: $predictionTime seconds")
    predictions.printSchema()

    val selected = predictions.select("activityLabel", "binaryLabel", "features", "rawPrediction", "prediction")
    selected.show()

    val evaluator = SingletonEvaluator.getEvaluator

    val output = evaluator.evaluate(predictions)

    println(s"Classification results for $modelName: ")
    if (singleFileUsed)
      println(s"The accuracy of the model $modelName for input file $singleFileName using AUROC is: $output")
    else println(s"The accuracy of the model $modelName for input file $multipleFolder using AUROC is $output")

  }

  /**
    * Perform cross validation on the data and select the best pipeline model given the data and parameters
    * @param trainData the training dataset
    * @param testData the test dataset
    * @param params the parameters that can be set in the Pipeline - currently LogisticRegression only, may need to amend
    * @param pipeline the pipeline to which cross validation is being applied
    * @param elm the ELM model being cross-validated
    */
  private def performCrossValidation(trainData: DataFrame, testData: DataFrame, params: DefaultELMParams, pipeline: Pipeline, elm: ELMClassifier) :Unit = {

    val paramGrid = new ParamGridBuilder()
      .addGrid(elm.activationFunc, Array(params.activationFunc, "tanh", "sin"))
      .addGrid(elm.hiddenNodes, Array(params.hiddenNodes, 10, 100, 200))
      .build()

    println(s"ParamGrid size is ${paramGrid.size}")

    val evaluator = SingletonEvaluator.getEvaluator

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val cvStartTime = System.nanoTime()
    val cvModel = cv.fit(trainData)
    val cvPredictions = cvModel.transform(testData)
    cvPredictions.select("activityLabel", "binaryLabel", "features", "rawPrediction", "prediction").show

    evaluator.evaluate(cvPredictions)
    val crossValidationTime = (System.nanoTime() - cvStartTime) / 1e9
    println(s"Cross validation time: $crossValidationTime seconds")

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    val avgParams = cvModel.avgMetrics

    val bestELMParams = bestModel.stages.last.asInstanceOf[ELMModel].explainParams

    println(s"The best model is ${bestModel.toString()} and the params are $bestELMParams")
  }
}