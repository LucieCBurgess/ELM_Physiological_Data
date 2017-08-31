package dev.elm

import breeze.linalg.{pinv, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, Vector => BV}
import breeze.linalg.NumericOps
import dev.data_load.SparkSessionWrapper
import breeze.numerics._
import breeze.linalg.functions._
import org.apache.spark.sql.{DataFrame, Dataset}

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
sealed class ELMClassifierAlgo protected (val hiddenNodes: Int, val activationFunc: ActivationFunc = new ActivationFunc, val ds: Dataset[_])
  extends SparkSessionWrapper {

  //FIXME activationFunc needs own class but a String and simple function will do for now
  // Need to convert ds$(features) to a BDM
  private val X: BDM[Double] = featureCols.inputMatrix // each training sample of (attributes, label) out into an input matrix. X has N rows and (attributes + label) columns
  private val N: Int = X.rows
  val L: Int = hiddenNodes
  private val H: BDM[Double] = BDM.zeros[Double](N, L)
  val T = labelCol.inputMatrix
  private val beta: BDV[Double] = BDV.zeros[Double](L)


  /** select activationFunc based on input String */
  object activationFuncSigmoid {
    def sigmoid(x: Double): Double = 1 / (1 + math.pow(math.E, -x))
  }

  class ActivationFunc {
    def sigmoid: Double => Double = activationFuncSigmoid.sigmoid //maps doubles for all values of x
  }

  /** Step 1: randomly assign input weight w and bias b */
  private val w: BDM[Double] = BDM.rand[Double](L, X.cols)

  private val bias: BDV[Double] = BDV.rand[Double](L)

  def train(): BDV[Double] = {

    /** Step2: calculate the hidden layer output matrix H */
     for (i <- 0 until N)
       for (j <- 0 until L)
         H(i,j) = activationFuncSigmoid(w(j, ::) * X(i, ::).t + bias(j))
    
  }

}

