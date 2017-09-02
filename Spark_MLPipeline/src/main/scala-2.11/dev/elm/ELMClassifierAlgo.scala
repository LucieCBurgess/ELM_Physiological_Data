package dev.elm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.ml.linalg.{Vector, DenseVector => SDV}
import breeze.linalg.pinv
import dev.data_load.SparkSessionWrapper
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.Dataset

/**
  * Created by lucieburgess on 31/08/2017.
  * This class contains the logic of the learning algorithm Extreme Learning Machine.
  * Keeping the class separate to keep the code clean.
  * Will require a new instance of this to be instantiated within ELMClassifier.train()
  * hiddenNodes and activationFunc need to be passed in from ELMClassifier
  * Class defined as sealed to prevent other classes and objects extending ELMClassifierAlgo
  * This function calculates the output weight matrix beta and passes it to transform() in ClassifierModel
  * X is a matrix of the feature columns. X.rows (in Breeze) is the number of rows  which is the number of training samples.
  * X.cols (in Breeze) is the number of columns which is the same as the number of features.
  * Extends ELMClassifier so we have access to the parameters featuresCol, labelCol
  */

//FIXME don't want to pass the hidden nodes or activation func through to the Algo?? Have already been set as parameters in Classifier
sealed class ELMClassifierAlgo (val ds: Dataset[_]) extends ELMClassifier with SparkSessionWrapper {

  import spark.implicits._

  private val X: BDM[Double] = computeX(ds) //features matrix
  private val N: Int = X.rows // Number of training samples
  private val L: Int = getHiddenNodes // Number of hidden nodes, parameter set in ELMClassifier
  private val H: BDM[Double] = BDM.zeros[Double](N, L) //Hidden layer output matrix, initially empty
  private val T: BDV[Double] = computeT(ds) //labels vector
  private val chosenAF = new ActivationFunction(getActivationFunc) //activation function, set in ELMClassifier

  /** Step 1: randomly assign input weight w and bias b */
  val w: BDM[Double] = BDM.rand[Double](L, X.cols)

  val bias: BDV[Double] = BDV.rand[Double](L)

  /** Calculates the output weight vector beta of length L where L is the number of hidden nodes*/
  def calculateBeta(): BDV[Double] = {

    /** Step2: calculate the hidden layer output matrix H */
    for (i <- 0 until N)
      for (j <- 0 until L)
        H(i, j) = chosenAF.function(w(j, ::) * X(i, ::).t + bias(j))

    /** Step 3: Calculate the output weight beta. Column vector of length L */
    val beta = pinv(H) * T // check this is of the same rank as BDV.zeros[Double](L)
    beta
  }

  /** Calculates label predictions for this model */
  //val predictedLabels: SDV = this.predictAllLabels(X)

  /**
    * Predicts label for a single training sample
    * //FIXME - need to work out how to extract a single row from X. Grrr!
    * @param feature the vector of features for a single training sample (a row).
    * @param beta the output weights calculated in training the model
    * @return the raw predicted label - needs checking!!
    */
  def predictLabel(feature: BDV[Double], beta: BDV[Double]): Double = {
    val node: IndexedSeq[Double] = for (i <- 0 until L) yield (beta(i) * chosenAF.function(w(i, ::) * feature + bias(i)))
    node.sum.round.toDouble
  }

  /**
    * Predicts labels vector from the input features matrix, X
    * @param beta the output weights calculated by the model
    * @return SDV of predicted labels from an input dataset of features.
    */
  def predictAllLabels(beta: BDV[Double]): SDV = {
    val predictedLabels = new Array[Double](N)
    for (i <- 0 until N) {
      predictedLabels(i) = predictLabel(X(i, ::).t, beta)
    }
    new SDV(predictedLabels)
  }

  /** Need to test this function, not convinced it's working */
  // Returns a labels vector of length N
  // Bit confused about whether we need to generate the bias again ... check this in test. Presumably not since L is a constant,
  // The number of hidden nodes
  def predictAllLabelsInOneGo(ds: Dataset[_], beta: BDV[Double]) :Vector = {

    val datasetX: BDM[Double] = computeX(ds)
    //val datasetL: Int = beta.length
    val numSamples:Int = datasetX.rows //N = numSamples for the dataset
    val predictedLabels = new Array[Double](numSamples)
    for (i <- 0 until numSamples) { // for i <- 0 until N, for j <- 0 until L
      val node: IndexedSeq[Double] = for (j <- 0 until L) yield (beta(j) * chosenAF.function(w(j, ::) * datasetX(i, ::).t + bias(j)))
      predictedLabels(i) = node.sum.round.toDouble
    }
    new SDV(predictedLabels)
  }


  // ******************************** Data-wrangling helper functions *******************************
  /**
    * Helper function to compute an array from the dataset features
    * @param ds the dataset to be operated upon
    * @return underlying array of values, in the form of Doubles.
    */
  private def computeFeaturesArray(ds: Dataset[_]) :Array[Double] = {
    val array: Array[Double] = ds.select(col($(featuresCol))).as[Double].collect()
    array
  }

  /**
    * Helper function to select features from a Spark dataset and return as a Breeze Dense Matrix. This allows pinv to be used
    * @param ds the dataset being operated on, used in the ELMClassifier class
    * @return X, a BreezeDenseMatrix of the dataset feature values, to be used in the ELM algorithm above
    */
  private def computeX(ds: Dataset[_]): BDM[Double] = {

    //val array2: Array[Double] = ds.select("features").toDF().collect.map(_.getDouble(0))
    val array: Array[Double] = ds.select(col($(featuresCol))).as[Double].collect()
    val numFeatures: Int = ds.select(col($(featuresCol))).head.get(0).asInstanceOf[Vector].size
    new BDM(ds.count.toInt, numFeatures, array)
  }

  /**
    * Helper function to select labelCol from a Spark dataset and return it as a Breeze Dense Vector. This allows pinv to be used
    * @param ds the dataset being operated on, used in the ELMClassifier class
    * @return T, a BreezeDenseVector of the dataset label values, to be used in the ELM algorithm above
    */
  private def computeT(ds: Dataset[_]): BDV[Double] = {

    //val array3: Array[Double] = ds.select(col("labels")).toDF().collect().map(_.getDouble(0))
    val array: Array[Double] = ds.select(col($(labelCol))).as[Double].collect()
    new BDV(array)
  }
}




