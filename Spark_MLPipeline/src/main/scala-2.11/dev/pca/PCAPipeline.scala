package dev.pca

import dev.logreg.{LRParams, SparkSessionWrapper}
import dev.data_load.{DataLoad, MHealthUser}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator}
import org.apache.spark.ml.feature.{VectorAssembler, PCA}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage, Transformer}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

import scala.collection.mutable

/**
  * Created by lucieburgess on 01/09/2017.
  * Added some additional code to the LRPipeline to play with PCA (Principal Components Analysis)
  */
object PCAPipeline extends SparkSessionWrapper {

    import spark.implicits._

    val fileName: String = "mHealth_subject1.txt"

    val defaultParams = LRParams()

    def main(args: Array[String]) {

      println(s"PCA pipeline component added to LR model: \n$defaultParams")

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

      val Array(trainData, testData) = data.randomSplit(Array(0.5,0.5),seed = 12345)

      data.show()

      /**
        * Combine columns which we think will predict Activity into a single feature vector
        */
      val featureCols: Array[String] = ScalaReflection.schemaFor[MHealthUser].dataType match {
        case s: StructType => s.fieldNames
        case _ => Array[String]()
      }
      val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
      pipelineStages += featureAssembler

      val pca = new PCA()
          .setInputCol("features")
        .setOutputCol("pcaFeatures")
        .setK(3)
        .fit(data)

      val result = pca.transform(data).select("pcaFeatures")

      println("*************** Printing the principal components *****************************")
      result.show()

      /** Create the classifier, set parameters for training */
      val lr = new LogisticRegression()
        .setFeaturesCol("pcaFeatures")
        .setRegParam(0.0)
        .setElasticNetParam(0.0)
        .setLabelCol("binaryLabel")
        .setMaxIter(100)
        .setFitIntercept(true)

      pipelineStages += lr
      println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

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
      println("************ Evaluating model and calculating train and test areaUnderROC - larger is better ************")
      evaluateClassificationModel("Train",pipelineModel, trainData)
      evaluateClassificationModel("Test", pipelineModel, testData)

      /** Perform cross-validation on the dataset */
      // FIXME - how to choose different features and change these parameters
      println("************* Performing cross validation and computing best parameters ************")
      performCrossValidation(trainData, testData, defaultParams, pipeline, lr)

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

      val selected = predictions.select("activityLabel", "binaryLabel", "modelFeatures", "rawPrediction", "probability", "prediction")
      selected.show()

      val evaluator = SingletonEvaluator.getEvaluator

      val output = evaluator.evaluate(predictions)

      println(s"Classification results for $modelName: ")
      println(s"The accuracy of the model $modelName for input file $fileName using areaUnderROC is: $output")

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
      cvPredictions.select("activityLabel", "binaryLabel", "modelFeatures", "rawPrediction", "probability", "prediction").show

      evaluator.evaluate(cvPredictions)
      val crossValidationTime = (System.nanoTime() - cvStartTime) / 1e9
      println(s"Cross validation time: $crossValidationTime seconds")

      val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
      val avgParams = cvModel.avgMetrics

      val bestLRParams = bestModel.stages.last.asInstanceOf[LogisticRegressionModel].explainParams

      println(s"The best model is ${bestModel.toString()} and the params are $bestLRParams")

    }

}



