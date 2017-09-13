package dev.logreg

/**
  * Created by lucieburgess on 15/08/2017.
  * Logistic Regression pipeline following numerous online examples to learn Spark programming constructs and apply pipeline
  * model to ELM
  * Main method is in LRTestMain.scala
  */

import dev.data_load.{DataLoadOption, DataLoadWTF}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage, Transformer}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import scala.collection.mutable

object LRPipeline {

  /** Amend singleFileUsed to false if more than a single file is being used
    * Amend "Multiple3" to "Multiple10" if running pipeline across full dataset of 10 users
    */
  val singleFileName: String = "mHealth_subject1.txt"
  val singleFileUsed: Boolean = false
  val multipleFolder: String = "Multiple3"

  def run(params: LRParams) :Unit = {

    val spark: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName(s"Logistic Regression example with $params").getOrCreate()

    import spark.implicits._

    println(s"Logistic Regression applied to the mHealth data with parameters: \n$params")

    /** Load training and test data and cache it */
    val df2 = if(singleFileUsed) DataLoadOption.createDataFrame(singleFileName) match {
      case Some(df) => df
      case None => throw new UnsupportedOperationException("Couldn't create DataFrame")
      }
      else DataLoadWTF.createDataFrame(multipleFolder)

    val df3 = df2.filter($"activityLabel" > 0)
        .withColumn("binaryLabel", when($"activityLabel".between(1.0, 3.0), 0.0).otherwise(1.0))
        .withColumn("uniqueID", monotonically_increasing_id())

    /** Set up the pipeline stages */
    val pipelineStages = new mutable.ArrayBuffer[PipelineStage]()

    df3.show()

    /**
      * Combine columns which we think will predict activity into a single feature vector
      * Sets the input columns and the output column as a new column, a vector, 'features' and adds to the pipeline
      */
    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z", "acc_Arm_X", "acc_Arm_Y", "acc_Arm_Z")
    val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    pipelineStages += featureAssembler

    /** Create the classifier, set parameters for training */
    val lr = new LogisticRegression()
        .setFeaturesCol("features")
        .setLabelCol("binaryLabel")
        .setRegParam(params.regParam)
        .setElasticNetParam(params.elasticNetParam)
        .setMaxIter(params.maxIter)
        .setTol(params.tol)
        .setFitIntercept(params.fitIntercept)

    pipelineStages += lr
    println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

    /** Randomly split data into test, train with 50% split */
    val train: Double = 1-params.fracTest
    val test: Double = params.fracTest
    val Array(trainData, testData) = df3.randomSplit(Array(train, test), seed = 12345)

    println(s"The size of the training set is ${trainData.count()}")
    println(s"The size of the test set is ${testData.count()}")

    /** Set the pipeline from the pipeline stages */
    val pipeline: Pipeline = new Pipeline().setStages(pipelineStages.toArray)

    /** Fit the pipeline, which includes training the model */
    val startTime = System.nanoTime()
    val pipelineModel: PipelineModel = pipeline.fit(trainData)
    val trainingTime = (System.nanoTime() - startTime) / 1e9
    println(s"Training time: $trainingTime seconds")

    /** Print the weights and intercept for logistic regression, from the trained model */
    val lrModel = pipelineModel.stages.last.asInstanceOf[LogisticRegressionModel]
    println(s"Training features are as follows: ")
    println(s"Weights: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    /** Make predictions and evaluate the model using BinaryClassificationEvaluator */
    println("Evaluating model and calculating train and test AUROC - larger is better")
    evaluateClassificationModel("Train",pipelineModel, trainData)
    evaluateClassificationModel("Test", pipelineModel, testData)

    /** Perform cross-validation on the dataset */
    println("Performing cross validation and computing best parameters")
    performCrossValidation(trainData, testData, params, pipeline, lr)

    spark.stop()
  }

  /**
    * Singleton version of BinaryClassificationEvaluator so we can use the same instance across the whole model
    *  metric must be "areaUnderROC" or "areaUnderPR" according to BinaryClassificationEvaluator API
    */
  private object SingletonEvaluator {
      val evaluator: Evaluator = new BinaryClassificationEvaluator()
        .setMetricName("areaUnderROC")
        .setLabelCol("binaryLabel")
        .setRawPredictionCol("rawPrediction")

      def getEvaluator: Evaluator = evaluator
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

    val selected = predictions.select("activityLabel", "binaryLabel", "features", "rawPrediction", "probability", "prediction")
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
    * @param lr the LogisticRegression model being cross-validated
    */
  private def performCrossValidation(trainData: DataFrame, testData: DataFrame, params: LRParams, pipeline: Pipeline, lr: LogisticRegression) :Unit = {

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(params.regParam, 0.01, 0.1))
      .addGrid(lr.maxIter, Array(10, 50, params.maxIter))
      .addGrid(lr.elasticNetParam, Array(params.elasticNetParam, 0.5, 1.0))
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
    cvPredictions.select("activityLabel", "binaryLabel", "features", "rawPrediction", "probability", "prediction").show

    evaluator.evaluate(cvPredictions)
    val crossValidationTime = (System.nanoTime() - cvStartTime) / 1e9
    println(s"Cross validation time: $crossValidationTime seconds")

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    val avgParams = cvModel.avgMetrics

    val bestLRParams = bestModel.stages.last.asInstanceOf[LogisticRegressionModel].explainParams

    println(s"The best model is ${bestModel.toString()} and the params are $bestLRParams")

  }
}


