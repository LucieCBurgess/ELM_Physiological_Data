package dev.elm

import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.ml.linalg.{Vector, DenseMatrix => SDM, DenseVector => SDV}
import breeze.linalg.{*, pinv, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, Vector => BV}
import breeze.numerics._
import dev.data_load.SparkSessionWrapper
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

/**
  * Created by lucieburgess on 27/08/2017.
  * This class is a Transformer according to the Spark Pipeline model, and extends ClassificationModel
  * The Classifier is trained with ELMClassifier from the algorithm ELMClassifierAlgo
  *
  * This uses the default implementation of transform(), which reads column "features" and
  * outputs columns "prediction". 'rawPrediction" is calculated for each labelled point in the dataset
  * from the parameters passed into ELMClassifier, based on the ELMClassifier trained using column "features".
  * It also uses the default implementation of transformSchema(schema: StructType).
  *
  * This uses the default implementation of predict(), which chooses the label corresponding to
  * the maximum value returned by [[predictRaw()]], as stated in the Classifier Model API
  *
  */
class ELMModel(override val uid: String, val modelWeights: BDM[Double], val modelBias: BDV[Double], val modelBeta: BDV[Double], val modelHiddenNodes: Int, val modelAF: ActivationFunction)
  extends ClassificationModel[Vector, ELMModel] with SparkSessionWrapper
    with ELMParams with DefaultParamsWritable {

  import spark.implicits._

  override def copy(extra: ParamMap): ELMModel = {
    val copied = new ELMModel(uid, modelWeights, modelBias, modelBeta, modelHiddenNodes, modelAF)
    copyValues(copied, extra).setParent(parent)
  }

  /** Number of classes the label can take. 2 indicates binary classification */
  override val numClasses: Int = 2

  /**
    * @param features, the vector of features being input into the model
    * @return vector where element i is the raw prediction for label i. This raw prediction may be any real number,
    *         where a larger value indicates greater confidence for that label
    *         The underlying method in ClassificationModel then predicts raw2prediction, which given a vector of raw predictions
    *         selects the predicted labels. raw2prediction can be overridden to support thresholds which favour particular labels.
    *         NB. data.select("features") gives an instance of a dataframe, so this is the type of the features column
    */
    //FIXME - re-write this function to take a Vector parameter
    def predictRaw(features: Vector): SDV = {

    //val featuresArray: Array[Double] = features.rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
      val featuresArray: Array[Double] = features.toArray
      val featuresMatrix = new BDM[Double](numFeatures, features.size, featuresArray)

      val bias: BDV[Double] = modelBias // L x 1
      val weights: BDM[Double] = modelWeights //  L x numFeatures
      val beta: BDV[Double] = modelBeta // (L x N) . N => gives vector of length L

      val M = weights * featuresMatrix // L x numFeatures. numFeatures x N where N is no. of test samples. NB Features must be of size (numFeatures, N)
      val H = sigmoid((M(::, *)) + bias) // L x numFeatures
      val T = beta.t * H // L.(L x N) of type Transpose[DenseVector]
      new SDV((T.t).toArray) //length N
    }

}
