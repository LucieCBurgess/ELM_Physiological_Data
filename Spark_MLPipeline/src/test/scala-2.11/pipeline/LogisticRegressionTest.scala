package pipeline

/**
  * Created by lucieburgess on 15/08/2017.
  * Simple pipeline following numerous online examples to get the hang of things
  */

import data_load.{DataLoadTest, SparkSessionTestWrapper}
import scala.collection.mutable
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage, Transformer}
import org.apache.spark.sql.functions.when
import org.apache.spark.sql.functions._


object LogisticRegressionTest extends SparkSessionTestWrapper {

  //Main method moved to separate class, LRTestMain
  val fileName: String = "mHealth_subject1.txt"

  def run(params: LRTestParams) :Unit = {

    //val spark = SparkSession.builder().appName(s"LogisticRegressionTest with $params").master("local[*]").getOrCreate()

    import spark.implicits._

    println(s"Logistic Regression Example from the Spark examples with some dummy data and parameters: \n$params")

    /** Load training and test data and cache it */
    val data = DataLoadTest.createDataFrame(fileName)

    /** Set up the pipeline stages */
    val pipelineStages = new mutable.ArrayBuffer[PipelineStage]()

    /** Filter for unLabelled data and add a new binary column, indexedLabel */
      //FIXME how to add this step to the pipeline as a Transformer

    //val binarizedData :Transformer = new InitialTransformer() //Moved to a separate class, Transformer is abstract
      val df2 = data
        .filter($"activityLabel" > 0)
        .withColumn("binaryLabel",when($"activityLabel".between(1, 3), 0).otherwise(1))
        .withColumn("uniqueID", monotonically_increasing_id())

    /** Randomly split data into test, train with 50% split */
      // FIXME - add this to params. See DecisionTreeExample. Basically need to amend DataLoad method to include params
    val Array(trainData, testData) = df2.randomSplit(Array(0.5,0.5))

    //FIXME - in the real pipeline we would include a Feature Transformer step here to calculate velocity from accelerometer data

    /** Combine columns which we think will predict Activity into a single feature vector
      * In this simple example we will just include a few features as a proof of concept
      */
    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z", "acc_Arm_X", "acc_Arm_Y", "acc_Arm_Z")

    /** Sets the input columns as the array of features and the output column as a new column, model_features */
    val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("modelFeatures")
    pipelineStages += featureAssembler

    //Deleted StringIndexer, I don't think we need it

    /** Create the classifier, set parameters for training */
      // The parameters can alternatively be assigned using a ParamMap instead of a case class. See ML pipeline api
    val lr = new LogisticRegression()
        .setFeaturesCol("modelFeatures")
        .setLabelCol("binaryLabel")
          .setRegParam(params.regParam)
      .setElasticNetParam(params.elasticNetParam)
      .setMaxIter(params.maxIter)
      .setTol(params.tol)
      .setFitIntercept(params.fitIntercept)
    pipelineStages += lr
    println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

    /** Set the pipeline from the pipeline stages */
    val pipeline = new Pipeline().setStages(pipelineStages.toArray)

    /** Fit the pipeline */
    val startTime = System.nanoTime()
    val pipelineModel: PipelineModel = pipeline.fit(trainData) //trains the model
    val trainingTime = (System.nanoTime() - startTime) / 1e9
    println(s"Training time: $trainingTime seconds")

    /** Print the weights and intercept for logistic regression, from the trained model */
    val lrModel = pipelineModel.stages.last.asInstanceOf[LogisticRegressionModel]
    println(s"Training features are as follows: ")
    println(s"Weights: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    /** Make predictions and evaluate the model using BinaryClassificationEvaluator */
    evaluateClassificationModel("Train",pipelineModel, trainData)
    evaluateClassificationModel("Test", pipelineModel, testData)

  }

  /**
    * Evaluate the given ClassificationModel on data. Print the results. Based on decision tree example
    * @param model  Must fit ClassificationModel abstraction, with Transformers and Estimators
    * @param df  DataFrame with "prediction" and labelColName columns
    * @param metric must be "areaUnderROC" or "areaUnderPR" according to BinaryClassificationEvaluator API
    */
  private def evaluateClassificationModel(modelName: String, model: Transformer, df: DataFrame, metric: String = "areaUnderROC",
   labelCol: String = "binaryLabel", rawPredictionCol: String = "rawPrediction"): Unit = {

    val startTime = System.nanoTime()
    val predictions = model.transform(df).cache() // gives predictions for both training and test data
    val predictionTime = (System.nanoTime() - startTime) / 1e9
    println(s"Training time: $predictionTime seconds")
    predictions.printSchema()

    val selected = predictions.select("activityLabel", "binaryLabel", "modelFeatures", "rawPrediction", "probability", "prediction")
    selected.show()

    val evaluator = new BinaryClassificationEvaluator().setMetricName(metric).setLabelCol(labelCol).setRawPredictionCol(rawPredictionCol)

    val evaluatorParams = ParamMap(evaluator.metricName -> metric)

    val outputMetric = evaluator.evaluate(predictions, evaluatorParams)

    println(s"Classification results for $modelName: ")
    println(s"The accuracy of the model $modelName for input file $fileName using ${evaluator.getMetricName} is: $outputMetric")

  }
}


