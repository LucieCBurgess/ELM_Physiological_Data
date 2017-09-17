package dev.elm

import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.ml.linalg.{Vector, DenseVector => SDV}
import breeze.linalg.{*, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.DefaultParamsWritable

/**
  * Created by lucieburgess on 27/08/2017.
  * This class is a Transformer according to the Spark Pipeline model, and extends ClassificationModel
  * The Classifier is trained with ELMClassifier from the algorithm ELMAlgo
  *
  * predictRaw(features: Vector) :Vector is implemented. From this, "rawPrediction" is calculated for each labelled point
  * in the dataset, from the parameters passed into ELMClassifier, based on the ELMClassifier trained using column "features".
  *
  * This uses the default implementation of transform(), which reads column "features" and outputs columns "prediction".
  * It also uses the default implementation of transformSchema(schema: StructType).
  * It also uses the default implementation of predict(), which chooses the label corresponding to
  * the maximum value returned by predictRaw(), as stated in the Classifier Model API.
  */
class ELMModel(val uid: String, val modelWeights: BDM[Double], val modelBias: BDV[Double], val modelBeta: BDV[Double],
               val modelHiddenNodes: Int, val modelAF: String, val modelNumFeatures: Int)
  extends ClassificationModel[Vector, ELMModel]
    with ELMParams with DefaultParamsWritable {

  /**
    * Implements value in API class ml.Model
    * Copies extra parameters into the ParamMap
    */
  override def copy(extra: ParamMap): ELMModel = {
    val copied = new ELMModel(uid, modelWeights, modelBias, modelBeta, modelHiddenNodes, modelAF, modelNumFeatures)
    copyValues(copied, extra).setParent(parent)
  }

  /**
    * Implements value in ml.classification.ClassificationModel
    * Number of classes the label can take - 2 for binary classification
    */
  override val numClasses: Int = 2

  /** PredictionModel sets numFeatures default as -1 (unknown) so this overrides that value */
  override def numFeatures: Int = modelNumFeatures

  /**
    * @param features , the vector of features being input into the model
    * @return vector of predictions for a single sample in the input dataset,
    *         where element i is the raw prediction for label i. This raw prediction may be any real number,
    *         where a larger value indicates greater confidence for that label
    *         ClassificationModel API then predicts raw2prediction, which given a vector of raw predictions
    *         selects the predicted labels. raw2prediction can be overridden to support thresholds which favour particular labels.
    */
  override def predictRaw(features: Vector): Vector = {

    val featuresArray = features.toArray
    val featuresMatrix = new BDM[Double](modelNumFeatures, 1, featuresArray) //numFeatures x 1

    val bias: BDV[Double] = modelBias //L x 1
    val weights: BDM[Double] = modelWeights// Lx numFeatures
    val beta: BDV[Double] = modelBeta // (LxN).N => gives vector of length L

    val M = weights * featuresMatrix// L x numFeatures. numFeatures x 1 = L x 1

    val Z = M(::, *) + bias

    def calculateH(af: String, Z: BDM[Double]): BDM[Double] = af match {
      case "sigmoid" => sigmoid(Z)
      case "tanh" => tanh(Z)
      case "sin" => sin(Z)
      case _ => throw new IllegalArgumentException("Activation function must be sigmoid, tanh, or sin")
    }

    val H = calculateH(modelAF, Z) //1xL

    val T = beta.t * H // L.(L x 1) of type Transpose[DenseVector]

    val rawPredictions: Array[Double] = Array(-T(0), T(0))

    new SDV(rawPredictions) //length 2
  }
}
