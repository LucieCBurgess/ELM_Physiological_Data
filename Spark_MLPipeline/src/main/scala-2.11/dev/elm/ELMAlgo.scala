package dev.elm

import breeze.generic.{MappingUFunc, UFunc}
import breeze.linalg.{*, pinv, DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.ml.linalg.Vector
import breeze.numerics._
import org.apache.spark.sql.Dataset

/**
  * Created by lucieburgess on 31/08/2017.
  * This class contains the logic of the learning algorithm Extreme Learning Machine.
  * Various parameters are passed into ELM Model so that labels can be predicted from the parameters.
  * Class defined as sealed to prevent other classes and objects extending it.
  * This function calculates the output weight matrix beta and passes it to train in ELMClassifier and transform in ELMModel.
  * Parameters required by the class are ds, in the input dataset, the number of hidden nodes and the activation function.
  */
sealed class ELMAlgo(val ds: Dataset[_], hiddenNodes: Int, af: String) {

  import ds.sparkSession.implicits._

  /**
    * Step 1: calculate main variables used in the model
    * L is assigned to hiddenNodes to make matrix computation simpler to follow
    */
  val algoNumFeatures: Int = ds.select("features").head.get(0).asInstanceOf[Vector].size

  private val N: Int = ds.count().toInt

  private val L: Int = hiddenNodes

  /** Step 2: calculate features matrix, X from spark dataset
    * Features matrix is actually tranpose compared to the dataset features to preserve cardinality
    */
  private val X: BDM[Double] = extractFeaturesMatrix(ds)

  /** Step 3: extract the labels vector as a Breeze dense vector, length N */
  private val T: BDV[Double] = extractLabelsVector(ds)

  /** Step 4: randomly assign input weight matrix of size (L, numFeatures) */
  val weights: BDM[Double] = BDM.rand[Double](L, algoNumFeatures)

  /** Step 5: randomly assign bias column vector of length L */
  val bias: BDV[Double] = BDV.rand[Double](L)

  /**
    * Step 6: Calculate the output weight vector beta of length L where L is the number of hidden nodes
    * pinv is the Moore-Penrose Pseudo-Inverse of matrix H
    */
  def calculateBeta(): BDV[Double] = {

    val M: BDM[Double] = weights * X //L x numFeatures. numFeatures x N = L x N

    val Z: BDM[Double] = (M(::, *) + bias) // was .t

    def calculateH(af: String, Z: BDM[Double]): BDM[Double] = af match {
      case "sigmoid" => sigmoid(Z)
      case "tanh" => tanh(Z)
      case "sin" => sin(Z)
      case _ => throw new IllegalArgumentException("Activation function must be sigmoid, tanh, or sin")
    }

    val H = calculateH(af, Z)

    println(s"H size is ${H.rows} rows, ${H.cols} columns")

    val beta: BDV[Double] = pinv(H).t * T // Vector of length L

    println(s"beta size is ${beta.length} ")
    beta
  }

  /**
    * Helper function to select features from a Spark dataset and return as a Breeze Dense Matrix. This allows pinv to be used
    *
    * @param ds the dataset being operated on, used in the ELMClassifier class
    * @return X, a BreezeDenseMatrix of the dataset feature values, to be used in the ELM algorithm above
    * NB. This gives a matrix in column major order and because the features are organised as vectors in each element
    *         of the features column, we need to swap rows and columns. Therefore X has numFeatures rows and N(number of training samples)
    * columns. This effectively gives a amtrix which is transpose to the matrix we want. However the algorithm
    *         requires us to transpose it, so we avoid that step and simply use the extracted version.
    */
  private def extractFeaturesMatrix(ds: Dataset[_]): BDM[Double] = {

    val array = ds.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
    new BDM[Double](algoNumFeatures, N, array)
  }

  /**
    * Helper function to select labelCol from a Spark dataset and return it as a Breeze Dense Vector. This allows pinv to be used
    *
    * @param ds the dataset being operated on, used in the ELMClassifier class
    * @return T, a BreezeDenseVector of the dataset label values, to be used in the ELM algorithm above
    */
  private def extractLabelsVector(ds: Dataset[_]): BDV[Double] = {

    val array = ds.select("binaryLabel").as[Double].collect
    new BDV[Double](array)
  }
}

