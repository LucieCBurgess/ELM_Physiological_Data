package dev.elm

import breeze.linalg.{pinv, DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.ml.linalg.{Vector, DenseMatrix => SDM, DenseVector => SDV}
import org.apache.spark.sql.Dataset
import breeze.linalg._
import breeze.math._
import breeze.numerics._

/**
  * Created by lucieburgess on 03/09/2017.
  * Trying to avoid the Java heap OOM error.
  */
sealed class ELMClassifierAlgoSpark(ds: Dataset[_], hiddenNodes: Int, af: String) extends ELMClassifier {

  import ds.sparkSession.implicits._

  val X: SDM = computeX(ds) //features matrix
  val N: Int = ds.count().toInt // Number of training samples
  val numFeatures = 3 // Hard-coded for now, need to link to a parameter
  val L: Int = hiddenNodes // Number of hidden nodes, parameter set in ELMClassifier

  val chosenAF = new ActivationFunction(af) //activation function, set in ELMClassifier

  /** Step 1: randomly assign input weight w and bias b */

  val random = new java.util.Random()

  val weights: SDM = SDM.randn(L, numFeatures, random)

  println(s"*************** The number of rows for weights is ${weights.numRows} *****************")
  println(s"*************** The number of coumns for weights is ${weights.numCols} *****************")

  val bias: SDM = SDM.randn(L, 1, random)

  println(s"*************** The number of rows for bias is ${bias.numRows} *****************")
  println(s"*************** The number of coumns for bias is ${bias.numCols} *****************")

  /** Calculates the output weight vector beta of length L where L is the number of hidden nodes*/
  def calculateBeta(): BDM[Double] = {

    /** Step2: calculate the hidden layer output matrix H */
    val wxplusbArray = weights.multiply(X.transpose).toArray // w.x(transpose)

    val biasArray = bias.toArray // bias

    val WXBBreeze = new BDM[Double](L, N, wxplusbArray)

    val biasBreeze = new BDM[Double](L, 1, biasArray)


    val H: BDM[Double] =  sigmoid(WXBBreeze + biasBreeze) // will have to define separate uFuncs if this doesn't work

    println(s"*************** The number of rows for H is  ${H.rows} *****************")
    println(s"*************** The number of cols for H is  ${H.cols} *****************")

    /** Step 3: Calculate the output weight beta. Matrix of length L with 1 column */

    val pinvH: BDM[Double] = pinv(H)

    val T: BDM[Double] = computeTasBreeze(ds)

    val beta: BDM[Double] = pinvH * T
    beta
  }


  // ******************************** Data-wrangling helper functions *******************************

  /**
    * Helper function to select features from a Spark dataset and return as a Breeze Dense Matrix. This allows pinv to be used
    * @param ds the dataset being operated on, used in the ELMClassifier class
    * @return X, a BreezeDenseMatrix of the dataset feature values, to be used in the ELM algorithm above
    */
  private def computeX(ds: Dataset[_]): SDM = {

    val array = ds.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
    println(s"The size of the features array is ${array.length} ************* (should be 3 x 22 = 66")

    println(s"The number of features is $numFeatures ************* (should be 6)")
    println(s"The size of the SDM is $N rows, $numFeatures cols (should be 22 x 6")
    new SDM(N, numFeatures, array)
  }

  /**
    * Helper function to select labelCol from a Spark dataset and return it as a Breeze Dense Vector. This allows pinv to be used
    * @param ds the dataset being operated on, used in the ELMClassifier class
    * @return T, a BreezeDenseVector of the dataset label values, to be used in the ELM algorithm above
    */
  private def computeTasSpark(ds: Dataset[_]): SDV = {

    val array = ds.select("binaryLabel").rdd.map( r => r.getAs[Double](0)).collect
    new SDV(array)
  }

  /**
    * Helper function to select labelCol from a Spark dataset and return it as a Breeze Dense Matrix. This allows pinv to be used
    * @param ds the dataset being operated on, used in the ELMClassifier class
    * @return T, a BreezeDenseVector of the dataset label values, to be used in the ELM algorithm above
    */
  private def computeTasBreeze(ds: Dataset[_]): BDM[Double] = {

    val array = ds.select("binaryLabel").rdd.map( r => r.getAs[Double](0)).collect
    new BDM[Double](N, 1, array)
  }
}

//val array = ds.select("binaryLabel").as[Double].collect()
