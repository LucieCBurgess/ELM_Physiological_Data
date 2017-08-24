package logistic_regression

/**
  * Created by lucieburgess on 15/08/2017.
  */

import data_load.{DataLoadTest, SparkSessionTestWrapper}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}

import scala.collection.mutable
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineStage, Transformer}
import org.apache.spark.ml.util.MetadataUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions.when


object LogisticRegressionTest extends SparkSessionTestWrapper {

  case class Params(
                     dataFormat: String = "text converted to DataFrame",
                     regParam: Double = 0.0,
                     elasticNetParam: Double = 0.0,
                     maxIter: Int = 100,
                     fitIntercept: Boolean = true,
                     tol: Double = 1E-6,
                     fracTest: Double = 0.5) // extends AbstractParams[Params] - not required?

  def main(args: Array[String]): Unit = {

    /** Define a number of parameters for this logistic regression model using org.apache.spark.ml.classification.LogisticRegression
      * Set the layout for using the parsing the parameters using Scopt, https://github.com/scopt/scopt, simple command line options parsing
      */
    val defaultParams = Params()

    val parser = new scopt.OptionParser[Params]("Logistic Regression test using src/main/scala/org/apache/spark/examples/ml/LogisticRegressionExample.scala") {

      head("Logistic Regression parameters")

      opt[Double]("regParam").text(s"regularization parameter, default: ${defaultParams.regParam}")
        .action((x, c) => c.copy(regParam = x))

      opt[Double]("elasticNetParam").text(s"Elastic net parameter in range [0,1], default :${defaultParams.elasticNetParam}")
        .action((x, c) => c.copy(elasticNetParam = x))

      opt[Int]("maxIter").text(s"Maximum number of iterations for gradient descent, default: ${defaultParams.maxIter}")
        .action((x, c) => c.copy(maxIter = x))

      opt[Boolean]("fitIntercept").text(s"Fit intercept parameter, whether to fit an intercept, default: ${defaultParams.fitIntercept}")
        .action((x, c) => c.copy(fitIntercept = x))

      opt[Double]("tol").text(s"Parameter for the convergence tolerance for iterative algorithms, " +
        s" with smaller value leading to higher accuracy but greatest cost of more iterations, default: ${defaultParams.tol}")
        .action((x, c) => c.copy(tol = x))

      opt[Double]("fracTest").text(s"The fraction of the data to use for testing, default: ${defaultParams.fracTest}")
        .action((x, c) => c.copy(fracTest = x))

      opt[String]("dataFormat").text(s"Please enter the dataformat here, default: ${defaultParams.dataFormat}")
        .action((x, c) => c.copy(dataFormat = x))

      checkConfig { params =>
        if (params.fracTest < 0 || params.fracTest >= 1) {
          failure(s"fracTest ${params.fracTest} value is incorrect; it should be in range [0,1).")
        } else {
          success
        }
      }
    }

   parser.parse(args, defaultParams) match {
     case Some(params) => run (params)
     case _ => sys.exit(1)
   }

  }

  def run(params: Params) :Unit = {

    //val spark = SparkSession.builder().appName(s"LogisticRegressionTest with $params").master("local[*]").getOrCreate()

    import spark.implicits._

    println(s"Logistic Regression Example from the Spark examples with some dummy data and parameters: \n$params")

    /** Load training and test data and cache it */
    val data = DataLoadTest.createDataFrame("mHealth_subject1.txt")

    /** Filter for unLabelled data and add a new binary column, indexedLabel */
      //FIXME how to add this step to the pipeline as a Transformer
    val df2 = data
        .filter($"activityLabel" > 0)
        .withColumn("indexedLabel",when($"activityLabel".between(1, 3), 0).otherwise(1))

    //FIXME exception handling ... try ... catch block

    /** Randomly split data into test, train with 50% split */
      // FIXME - add this to params
    val Array(trainData, testData) = df2.randomSplit(Array(0.5,0.5))

    /** Set up the pipeline stages */
    val pipelineStages = new mutable.ArrayBuffer[PipelineStage]()

    //FIXME - in the real pipeline we would include a Feature Transformer step here to calculate velocity from accelerometer data

    /** Combine columns which we think will predict Activity into a single feature vector
      * In this simple example we will just include a few features as a proof of concept
      */
    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z", "acc_Arm_X", "acc_Arm_Y", "acc_Arm_Z")

    /** Sets the input columns as the array of features and the output column as a new column, model_features */
    val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("Model_features")
    pipelineStages += featureAssembler
    //val dataWithFeatures: DataFrame = featureAssembler.transform(data)
    // FIXME how to add this to the pipeline - do we need to ....
    //dataWithFeatures.show()

    /** Create a prediction column with StringIndexer for the output classifications */
    val labelIndexer = new StringIndexer().setInputCol("indexedLabel").setOutputCol("predictedLabel")
    pipelineStages += labelIndexer

    /** Create the classifier, set parameters for training */
    val lr = new LogisticRegression()
      .setFeaturesCol("Model_features")
      .setLabelCol("predictedLabel") // Label or predicted label?
      .setRegParam(params.regParam)
      .setElasticNetParam(params.elasticNetParam)
      .setMaxIter(params.maxIter)
      .setTol(params.tol)
      .setFitIntercept(params.fitIntercept)
    pipelineStages += lr

    /** Set the pipeline from the pipeline stages */
    val pipeline = new Pipeline().setStages(pipelineStages.toArray)

    /** Fit the pipeline */
    val startTime = System.nanoTime()
    val pipelineModel = pipeline.fit(trainData)
    val elapsedTime = (System.nanoTime() - startTime) / 1e9
    println(s"Training time: $elapsedTime seconds")

    val lrModel = pipelineModel.stages.last.asInstanceOf[LogisticRegressionModel]

    /** Print the weights and intercept for logistic regression */
    println(s"Weights: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    println("Training data results:")
    evaluateClassificationModel(pipelineModel, trainData, "indexedLabel")
    println("Test data results:")
    evaluateClassificationModel(pipelineModel, testData, "indexedLabel")

    spark.stop()

    // Probably mixing metaphors here - need to follow the simple breast cancer example first and then transform this to a pipeline
    // See https://mapr.com/blog/predicting-breast-cancer-using-apache-spark-machine-learning-logistic-regression/

  }

  /**
    * Evaluate the given ClassificationModel on data. Print the results. Based on decision tree example
    * @param model  Must fit ClassificationModel abstraction, with Transformers and Estimators
    * @param df  DataFrame with "prediction" and labelColName columns
    * @param labelColName  Name of the labelCol parameter for the model, i.e. name of the column which contains the label for the model
    *
    */
  private def evaluateClassificationModel(model: Transformer, df: DataFrame, labelColName: String): Unit = {

    val fullPredictions = model.transform(df).cache()
    // FIXME java.lang.ClassCastException: java.lang.Integer cannot be cast to java.lang.Double line 184
    val labels = fullPredictions.select(labelColName).rdd.map(_.getDouble(0))
    val predictions = fullPredictions.select("predictedLabel").rdd.map(_.getDouble(0))

    // FIXME - error MetadataUtils is not accessible from this place
    // Print number of classes for reference.
    //val numClasses = MetadataUtils.getNumClasses(fullPredictions.schema(labelColName)) match {
    //  case Some(n) => n
    //  case None => throw new RuntimeException(
    //    "Unknown failure when indexing labels for classification.")
    //}
    val accuracy = new MulticlassMetrics(predictions.zip(labels)).accuracy
    println(s" Accuracy $accuracy")
    // println(s"  Accuracy ($numClasses classes): $accuracy")
  }


}


