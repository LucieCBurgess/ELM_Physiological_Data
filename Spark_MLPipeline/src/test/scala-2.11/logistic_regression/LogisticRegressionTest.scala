package logistic_regression

/**
  * Created by lucieburgess on 15/08/2017.
  */

import data_load.mHealthUser
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import scala.collection.mutable
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.{PipelineStage, Pipeline}

object LogisticRegressionTest {

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

  } // end of main method

  def run(params: Params) :Unit = {

    val spark = SparkSession.builder().appName(s"LogisticRegressionTest with $params").master("local[*]").getOrCreate()

    import spark.implicits._

    println(s"Logistic Regression Example from the Spark examples with some dummy data and parameters: \n$params")

    //Load training and test data and cache it

    val data = spark.sparkContext
      .textFile("/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/mHealth_subject1.txt")
      .map(_.split("\\t"))
      .map(attributes => mHealthUser(attributes(0).toDouble, attributes(1).toDouble, attributes(2).toDouble,
        attributes(3).toDouble, attributes(4).toDouble,
        attributes(5).toDouble, attributes(6).toDouble, attributes(7).toDouble,
        attributes(8).toDouble, attributes(9).toDouble, attributes(10).toDouble,
        attributes(11).toDouble, attributes(12).toDouble, attributes(13).toDouble,
        attributes(14).toDouble, attributes(15).toDouble, attributes(16).toDouble,
        attributes(17).toDouble, attributes(18).toDouble, attributes(19).toDouble,
        attributes(20).toDouble, attributes(21).toDouble, attributes(22).toDouble,
        attributes(23).toInt))
      .toDF()
      .filter($"activityLabel" > 0)
      .withColumn("Label", $"(when activityLabel > 4, 1).otherwise(0)") //FIXME need to unit test this line
      .cache()

    val Array(trainData, testData) = data.randomSplit(Array(0.5,0.5))

    /** Set up the pipeline */
    val pipeline = new mutable.ArrayBuffer[PipelineStage]()


    //FIXME - in the real pipeline we would include a Feature Transformer step here to calculate velocity from accelerometer data

    /** Combine columns which we think will predict Activity into a single feature vector
      * In this simple example we will just include a few features as a proof of concept
      */
    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z", "acc_Arm_X", "acc_Arm_Y", "acc_Arm_Z")

    /** Sets the input columns as the array of features and the output column as a new column, model_features */
    val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("Model_features")
    pipeline += featureAssembler
    val dataWithFeatures: DataFrame = featureAssembler.transform(data) // FIXME how to add this to the pipeline
    dataWithFeatures.show()

    /** Create a prediction column with StringIndexer for the output classifications */
    val labelIndexer = new StringIndexer().setInputCol("Label").setOutputCol("Prediction")

    /** Create the classifier, set parameters for training */



    //Set up the pipeline - stage1 and stage2 are val names of the pipeline stages
    // val pipeline = new Pipeline().setStages(stage1, stage2)


  }



  }


}
