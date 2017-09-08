package dev.elm

import breeze.linalg.{*, pinv, DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.ml.linalg.{Vector, DenseVector => SDV}
import breeze.numerics.sigmoid
import dev.data_load.SparkSessionWrapper
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.Dataset

/**
  * Created by lucieburgess on 31/08/2017.
  * This class contains the logic of the learning algorithm Extreme Learning Machine.
  * Various parameters are passed into ELM Model so that labels can be predicted from the parameters.
  * Class defined as sealed to prevent other classes and objects extending ELMClassifierAlgo
  * This function calculates the output weight matrix beta and passes it to transform() in ClassifierModel.
  * This takes a parameter, ds, which is the input training set to the model.
  */

sealed class ELMAlgo(val ds: Dataset[_], hiddenNodes: Int, af: String) { // previously ExtendsELMClassifier

  import ds.sparkSession.implicits._

  /** Step 1: calculate main variables used in the model */
  val numFeatures: Int = ds.select("features").head.get(0).asInstanceOf[Vector].size
  println(s"The number of features is $numFeatures")

  //FIXME - not currently used, as using the sigmoid and tanh built in Breeze functions.
  val chosenAF = new ActivationFunction(af) //activation function, set in ELMClassifier

  private val N: Int = ds.count().toInt
  println(s"The number of training samples is $N")

  private val L: Int = hiddenNodes // Number of hidden nodes, parameter set in ELMClassifier
  println(s"The number of hidden nodes is $L")

  /** Step 2: calculate features matrix, X from spark dataset */
  private val X: BDM[Double] = extractFeaturesMatrix(ds) //features matrix, which is transpose to preserve cardinality

  /** Step 3: extract the labels vector as a Breeze dense vector */
  private val T: BDV[Double] = extractLabelsVector(ds) //labels vector

  private val H: BDM[Double] = BDM.zeros[Double](N, L) //Hidden layer output matrix, initially empty


  /** Step 4: randomly assign input weight matrix of size (L, numFeatures) */
  val weights: BDM[Double] = BDM.rand[Double](L, numFeatures) // L x numFeatures
  println(s"*************** The number of rows for weights is ${weights.rows} *****************")
  println(s"*************** The number of coumns for weights is ${weights.cols} *****************")

  /** Step 5: randomly assign bias vector of size L. */
  val bias :BDV[Double] = BDV.rand[Double](L) // bias is column vector of length L
  println(s"*************** The number of rows of the bias matrix is ${bias.length} *****************")

  /**
    * Step 6: Calculate the output weight vector beta of length L where L is the number of hidden nodes
    * pinv is the Moore-Penrose Pseudo-Inverse of matrix H
    */
  def calculateBeta(): BDV[Double] = {

    val M: BDM[Double] = weights * X //L x numFeatures. numFeatures x N = L x N

    val H: BDM[Double] = sigmoid((M(::,*) + bias).t) //N x L

    val beta: BDV[Double] = pinv(H) * T // Vector of length L

    println(s"*************** The number of rows for H is  ${H.rows} *****************")
    println(s"*************** The number of cols for H is  ${H.cols} *****************")
    println(s"*************** The length of Beta is  ${beta.length} *****************")

    beta
  }


  // ******************************** Data-wrangling helper functions *******************************

  /**
    * Helper function to select features from a Spark dataset and return as a Breeze Dense Matrix. This allows pinv to be used
    * @param ds the dataset being operated on, used in the ELMClassifier class
    * @return X, a BreezeDenseMatrix of the dataset feature values, to be used in the ELM algorithm above
    * NB. This gives a matrix in column major order and because the features are organised as vectors in each element
    * of the features column, we need to swap rows and columns. Therefore X has numFeatures rows and N(number of training samples)
    * columns. This effectively gives a amtrix which is transpose to the matrix we want. However the algorithm
    * requires up to tranpose it, so we don't need to do that, just use the extracted version.
    */
  private def extractFeaturesMatrix(ds: Dataset[_]): BDM[Double] = {

    val array = ds.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
    println(s"The size of the features matrix is $numFeatures rows, $N cols ***********")
    new BDM[Double](numFeatures,N,array)
  }

  /**
    * Helper function to select labelCol from a Spark dataset and return it as a Breeze Dense Vector. This allows pinv to be used
    * @param ds the dataset being operated on, used in the ELMClassifier class
    * @return T, a BreezeDenseVector of the dataset label values, to be used in the ELM algorithm above
    */
  private def extractLabelsVector(ds: Dataset[_]): BDV[Double] = {

    val array = ds.select("binaryLabel").as[Double].collect
    println(s"The size of the labels vector is ${array.length} ***********")
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

/**
  * Step 5: randomly assign bias vector of size L. Note we need this to be a matrix for later computations
  * made up of the same column repeated N times. This is not particularly efficient in terms of storage space.
  * However I can't find a way of adding a column vector repeatedly to different columns of a matrix.
  * I've tried this as a column slice in Breeze but it's not working so this is a work-around
  */
//  val biasVector: BDM[Double] = BDM.rand[Double](L, 1)
//  val biasArray: Array[Double] = biasVector.toArray // bias of Length L
//  val buf = scala.collection.mutable.ArrayBuffer.empty[Array[Double]]
//  for (i <- 0 until N) yield buf += biasArray
//  val replicatedBiasArray: Array[Double] = buf.flatten.toArray

