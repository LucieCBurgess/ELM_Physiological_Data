package dev.elm

import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * Created by lucieburgess on 16/09/2017.
  */
class ELMEvaluator(override val uid: String) extends Evaluator with ELMParams {

  override def copy(extra: ParamMap): ELMEvaluator = defaultCopy(extra)

  /**
    * Evaluates the training/test error on the input dataset, defined by the difference between the
    * actual label and the predicted label.
    * @param dataset
    * @return
    */
  override def evaluate(dataset: Dataset[_]) :Double = {

    import dataset.sparkSession.implicits._

    val difference: DataFrame = dataset.withColumn("squared_diff",(($"prediction" - $"binaryLabel") * ($"prediction" - $"binaryLabel")))
    val selected = difference.select("binaryLabel", "rawPrediction", "prediction", "squared_diff")
    selected.show()
    val squaredError = difference.select($"squared_diff").rdd.map(_(0).asInstanceOf[Double]).reduce(_+_)
    val squaredError2 = difference.select($"squared_diff").groupBy("squared_diff").sum()
    println("Showing the squared error")
    squaredError2.show()
    println(s"Evaluating the classifier accuracy: total number of misclassified observations is $squaredError")
    val meanError = squaredError / dataset.count.toDouble
    println(s"The training/test error as a % is: $meanError")
    meanError
  }

  def this() = this(Identifiable.randomUID("ELM_evaluator"))

  /**
    * Overrides method in
    * @return false as the best model is determined by the lowest mean test error
    */
  override def isLargerBetter: Boolean = false

}
