package prod.elm

import dev.data_load.{DataLoadOption, DataLoadWTF, MHealthUser}
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

  /**
    * Requires Spark driver size to be set to 8G for a single file for cross-validation on 3 folds.
    * Do not attempt to run this on more than a single dataset at one time.
    * The multiple file option is available for future development.
    * If using more than one file in future, set singleFileUsed to false.
    */
  val singleFileName: String = "mHealth_subject1.txt"
  val singleFileUsed: Boolean = true
  val multipleFolder: String = "Multiple3"

  def run(params: DefaultELMParams): Unit = {

    lazy val spark: SparkSession = {
      SparkSession.builder().master("local[*]").appName(s"ELM pipeline with $params").getOrCreate()
    }

    import spark.implicits._

    println(s"Extreme Learning Machine applied to the mHealth data with parameters: \n$params")

    /** Load training and test data and cache it */
    val df2 = if(singleFileUsed) DataLoadOption.createDataFrame(singleFileName) match {
      case Some(df) => df
      case None => throw new UnsupportedOperationException("Couldn't create DataFrame")
    }
    else DataLoadWTF.createDataFrame(multipleFolder)

    val data = df2.filter($"activityLabel" > 0)
      .withColumn("binaryLabel", when($"activityLabel".between(1.0, 3.0), 0.0).otherwise(1.0))
      .withColumn("uniqueID", monotonically_increasing_id())

    val Nsamples: Int = data.count().toInt
    println(s"The number of training samples is $Nsamples")

    /** Set up the pipeline stages */
    val pipelineStages = new mutable.ArrayBuffer[PipelineStage]()

    /** Combine columns which we think will predict labels into a single feature vector */
    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z")
    checkFeatureColsInSchema(featureCols)

    /**
      * Add the features to the DataFrame using VectorAssembler.
      * Pipeline.fit() does not pick up the VectorAssembler properly due to a bug in the API
      * so we have to transform the data outside the pipeline.
      */
    val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val dataWithFeatures = featureAssembler.transform(data)

    println(s"The schema of the MHealth data is: \n")
    dataWithFeatures.printSchema()

    /** Create the classifier, set parameters for training */
    val elm = new ELMClassifier()
      .setFeaturesCol("features")
      .setLabelCol("binaryLabel")
      .setHiddenNodes(params.hiddenNodes)
      .setActivationFunc(params.activationFunc)
      .setFracTest(params.fracTest)

    pipelineStages += elm
    println("ELM parameters:\n" + elm.explainParams() + "\n")

    /** Set the pipeline from the pipeline stages */
    val pipeline: Pipeline = new Pipeline().setStages(pipelineStages.toArray)

    /** Use fracTest parameter to set up the (trainData, testData) tuple and randomly split the data */
    val train: Double = 1 - elm.getFracTest
    val test: Double = elm.getFracTest
    val Array(trainData, testData) = dataWithFeatures.randomSplit(Array(train, test), seed = 12345)

    /** Fit the pipeline, which includes training the model */
    val startTime = System.nanoTime()

    val pipelineModel: PipelineModel = pipeline.fit(trainData)
    val elmModel = pipelineModel.stages.last.asInstanceOf[ELMModel]

    val trainingTime = (System.nanoTime() - startTime) / 1e9
    println(s"Training time: $trainingTime seconds")

    /** Make predictions and evaluate the model using BinaryClassificationEvaluator */
    println("Evaluating model, calculating training and test AUROC (larger is better) " +
      "and Confusion Matrix (smaller is better)")
    evaluateClassificationModel("Train",pipelineModel, trainData)
    evaluateClassificationModel("Test", pipelineModel, testData)

    /** Perform cross-validation on the dataset */
    println("Performing cross validation and computing best parameter using the Confusion Matrix approach:")
    performCrossValidation(testData, params, pipeline, elm)

    spark.stop()
  }

  /****************** Utilities for methods above *****************/

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
    *  .setRawPredictionCol expects a vector of length 2 so the predicted label must be output as a vector of
    *  (-rawPrediction, rawPrediction) for this to work.
    */
  private object AUROCSingletonEvaluator {
    val elmEvaluator: Evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setLabelCol("binaryLabel")
      .setRawPredictionCol("rawPrediction")

    def getEvaluator: Evaluator = elmEvaluator
  }

  /**
    * Simple custom evaluation of the model using ELMEvaluator, which compares the predictions to the labels and
    * computes the mean % of incorrectly classified labels across the whole dataset
    */
  private object ELMSingletonEvaluator {
    val elmNetEvaluator: Evaluator = new ELMEvaluator()
    def getElmNetEvaluator: Evaluator = elmNetEvaluator
  }

  /**
    * Evaluate the given ClassificationModel on data. Print the results
    * @param model  Must fit ClassificationModel abstraction, with Transformers and Estimators
    * @param df  DataFrame with "prediction" and labelColName columns
    */
  private def evaluateClassificationModel(modelName: String, model: Transformer, df: DataFrame): Unit = {

    val startTime = System.nanoTime()
    val predictions = model.transform(df).cache()
    val predictionTime = (System.nanoTime() - startTime) / 1e9
    println(s"Prediction time for model $modelName: is $predictionTime seconds")
    println(s"Printing the first 10 lines of the dataset $modelName with predictions")
    predictions.show(10)

    val aurocEvaluator = AUROCSingletonEvaluator.getEvaluator
    val output1 = aurocEvaluator.evaluate(predictions)

    val mseEvaluator = ELMSingletonEvaluator.getElmNetEvaluator
    val output2 = mseEvaluator.evaluate(predictions)

    println(s"Classification results for $modelName using AUROC (larger is better): ")
    if(singleFileUsed)
    println(s"The accuracy of the model $modelName for input file $singleFileName using AUROC is: $output1")
    else println(s"The accuracy of the model $modelName for input file $multipleFolder using AUROC is: $output1")

    println(s"Classification results for $modelName using ConfusionMatrix (smaller is better: ")
    if (singleFileUsed)
    println(s"The accuracy of the model $modelName for input file $singleFileName using Confusion Matrix is: $output2")
    else println(s"The accuracy of the model $modelName for input file $multipleFolder using Confusion Matrix is: $output2")
  }

  /**
    * Perform cross validation on the data and select the best pipeline model given the data and parameters
    * This model uses the confusion matrix/ mean-squared-error approach, not AUROC, as AUROC is not useful
    * for classifiers which do not assign a probability score to the predictions, such as neural networks
    * @param data the data for which cross validation is to be performed
    * @param params the default parameters set for the ELM.
    * @param pipeline the pipeline to which cross validation is being applied
    * @param elm the ELM model being cross-validated
    */
  private def performCrossValidation(data: DataFrame, params: DefaultELMParams, pipeline: Pipeline, elm: ELMClassifier) :Unit = {

    val paramGrid = new ParamGridBuilder()
      .addGrid(elm.activationFunc, Array(params.activationFunc, "tanh", "sin"))
      .addGrid(elm.hiddenNodes, Array(10, params.hiddenNodes, 100, 150))
      .build()

    println(s"ParamGrid size is ${paramGrid.size}")

    /** Note use of ELMSingletonEvaluator instead of BinarySingletonEvaluator */
    val evaluator = ELMSingletonEvaluator.getElmNetEvaluator

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2) //do not attempt to set this higher than 3 - it causes a JavaOOM error

    println("Performing cross-validation on the dataset")
    val cvStartTime = System.nanoTime()
    val cvModel = cv.fit(data)
    val cvPredictions = cvModel.transform(data)
    println(s"Parameters of this CV instance are ${cv.explainParams()}")

    evaluator.evaluate(cvPredictions)
    val crossValidationTime = (System.nanoTime() - cvStartTime) / 1e9
    println(s"Cross validation time: $crossValidationTime seconds")

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    val avgParams = cvModel.avgMetrics

    val bestELMParams = bestModel.stages.last.asInstanceOf[ELMModel].explainParams

    println(s"The best model is ${bestModel.toString()} and the params are $bestELMParams")
  }
}