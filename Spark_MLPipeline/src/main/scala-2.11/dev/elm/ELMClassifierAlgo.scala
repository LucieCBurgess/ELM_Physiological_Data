package dev.elm

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, Vector => BV}
import org.apache.spark.ml.linalg.{DenseVector => SDV, DenseMatrix => SDM}
import breeze.linalg.NumericOps
import breeze.linalg.pinv
import dev.data_load.SparkSessionWrapper
import breeze.numerics._
import breeze.linalg.functions._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.ml.

import scala.math.random


/**
  * Created by lucieburgess on 31/08/2017.
  * This class contains the logic of the learning algorithm based on the training data.
  * Keeping the class separate to keep the code clean.
  * Will require a new instance of this to be instantiated within ELMClassifier.train()
  * hiddenNodes and activationFunc need to be passed in from ELMClassifier
  */

/**
  * Class defined as sealed to prevent other classes and objects extending ELMClassifierAlgo
  * Object or sealed class?
  */
sealed class ELMClassifierAlgo protected (val ds: Dataset[_], val hiddenNodes: Int, val af: String)
  extends SparkSessionWrapper {

  //FIXME activationFunc needs own class but a String and simple function will do for now
  // Need to convert ds$(features) to a BDM
  private val X: BDM[Double] = inputMatrix.featureCols // put each training sample of (features, label) into an input matrix. X has N rows and (attributes + label) columns
  private val N: Int = X.rows
  private val L: Int = hiddenNodes
  private val H: BDM[Double] = BDM.zeros[Double](N, L)
  private val T: BDV[Double] = inputMatrix.labelCol
  //private val beta: BDV[Double] = BDV.zeros[Double](L)
  private val chosenAF = new ActivationFunction(af) // need to link this to the Parameters set at the beginning

  /** Step 1: randomly assign input weight w and bias b */
  private val w: BDM[Double] = BDM.rand[Double](L, X.cols)

  private val bias: BDV[Double] = BDV.rand[Double](L)

  def train(): BDV[Double] = {
    /** Step2: calculate the hidden layer output matrix H */
     for (i <- 0 until N)
       for (j <- 0 until L)
        H(i,j) = chosenAF.function(w(j, ::) * X(i, ::).t + bias(j))

    /** Step 3: Calculate the output weight beta. Column vector of length L */
   pinv(H) * T // check this is of the same rank as BDV.zeros[Double](L)
    //FIXME - how to convert beta to a matrix of coefficients, which is of the same size as the number of features?
    // Or is the coefficients just a hangover from the LogisticRegression example? Do we need to redefine transform() or predictRaw()?
  }

  /**
    * Helper function to extract featureCols from the input dataset and return as a BreezeDenseMatrix
    * First of all return to a SparkDenseMatrix then convert SDM to BDM and return the breeze version
    */
  private def computeX(ds: Dataset[_]) :BDM[Double] = {

    SDM.zeros(numRows: Int, numCols = Int)
    val X = SDM.zeros(ds.count().toInt, ds
  }

  /** Helper function to extract labelCol from the input dataset and return as a BreezeDenseVector */
  private def computeT(ds: Dataset[_]) :BDV[Double] = {
    ???
  }

  private def convertSparkDenseMatrixToBreezeDenseMatrix(m: SDM[Double]): BDM[Double] = {
    ???
  }

  private def convertSparkDenseVectorToBreezeDenseVector(v: SDV[Double]): BDV[Double] = {
    ???
  }
}




