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

sealed class ELMClassifierAlgo (val ds: Dataset[_], hiddenNodes: Int, af: String)
  extends ELMClassifier {

  import ds.sparkSession.implicits._

  private val X: BDM[Double] = extractFeaturesMatrixTranspose(ds) //features matrix
  private val N: Int = X.rows // Number of training samples
  private val L: Int = hiddenNodes // Number of hidden nodes, parameter set in ELMClassifier
  private val H: BDM[Double] = BDM.zeros[Double](N, L) //Hidden layer output matrix, initially empty
  private val T: BDV[Double] = extractLabelsVector(ds) //labels vector

  val chosenAF = new ActivationFunction(af) //activation function, set in ELMClassifier

  /** Step 1: randomly assign input weight w and bias b
    * weights is of rank (L x N)
    * Bias is of rank (L x 1)
    */

  val weights: BDM[Double] = BDM.rand[Double](L, X.cols)
  println(s"*************** The number of rows for w is ${weights.rows} *****************")
  println(s"*************** The number of coumns for w is ${weights.cols} *****************")

  val bias: BDM[Double] = BDM.rand[Double](L, 1)
  println(s"*************** The number of rows of the bias matrix is ${bias.rows} *****************")
  println(s"*************** The number of columns of the bias matrix is ${bias.cols} *****************")

  /** Calculates the output weight vector beta of length L where L is the number of hidden nodes*/
  def calculateBeta(): BDV[Double] = {

    val H = weights * X + bias

    val beta: BDV[Double] = pinv(H) * T

    println(s"*************** The number of rows for H is  ${H.rows} *****************")
    println(s"*************** The number of cols for H is  ${H.cols} *****************")

    beta
  }


//    /** Step2: calculate the hidden layer output matrix H */
//    for (i <- 0 until N)
//      for (j <- 0 until L)
//        H(i, j) = chosenAF.function(weights(j, ::) * X(i, ::).t + bias(j))

    /** Step 3: Calculate the output weight beta. Column vector of length L */
    //val pinvH = pinv(H)

    //val beta: BDV[Double] = pinv(H) * T // check this is of the same rank as BDV.zeros[Double](L) // Gives Out of Memory error
    // val beta = H / T // H is L x N so 10 x 35,000


  // ******************************** Data-wrangling helper functions *******************************

  /**
    * Helper function to select features from a Spark dataset and return as a Breeze Dense Matrix. This allows pinv to be used
    * @param ds the dataset being operated on, used in the ELMClassifier class
    * @return X, a BreezeDenseMatrix of the dataset feature values, to be used in the ELM algorithm above
    * //FIXME need to check the cardinality of the array compared to the features
    * This gives a
    */
  private def extractFeaturesMatrixTranspose(ds: Dataset[_]): BDM[Double] = {

    val array = ds.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
    println(s"The size of the features array is ${array.length} *************")
    val numFeatures: Int = ds.select("features").head.get(0).asInstanceOf[Vector].size
    println(s"The number of features is ${numFeatures} ************* (should be 3)")
    println(s"The size of the BDM is $numFeatures rows, $N cols ***********")
    new BDM[Double](numFeatures,N,array) //NB X gets transposed again above, so no need to transpose
  }

  /**
    * Helper function to select labelCol from a Spark dataset and return it as a Breeze Dense Vector. This allows pinv to be used
    * @param ds the dataset being operated on, used in the ELMClassifier class
    * @return T, a BreezeDenseVector of the dataset label values, to be used in the ELM algorithm above
    */
  private def extractLabelsVector(ds: Dataset[_]): BDV[Double] = {

    //val array = ds.select("binaryLabel").as[Double].collect()
    val array = ds.select("binaryLabel").rdd.map( r => r.getAs[Double](0)).collect
    new BDV[Double](array)
  }
}

/** Calculates label predictions for this model. Needs to be tested and moved to ELMModel */
//def predictAllLabelsInOneGo(ds: Dataset[_], beta: BDV[Double]) :Vector = {
//
//  val datasetX: BDM[Double] = computeX(ds)
//  val numSamples:Int = datasetX.rows //N
//  val predictedLabels = new Array[Double](numSamples)
//  for (i <- 0 until numSamples) { // for i <- 0 until N, for j <- 0 until L
//  val node: IndexedSeq[Double] = for (j <- 0 until L) yield (beta(j) * chosenAF.function(weights(j, ::) * datasetX(i, ::).t + bias(j)))
//  predictedLabels(i) = node.sum.round.toDouble
//}
//  new SDV(predictedLabels)
//}


