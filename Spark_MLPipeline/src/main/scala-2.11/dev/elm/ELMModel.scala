package dev.elm

import breeze.linalg.DenseVector
import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.ml.linalg.{BLAS, Vector, Vectors, DenseVector=>SparkDenseVector, DenseMatrix => SparkDenseMatrix, Matrices}
import org.apache.spark.ml.param.{IntParam, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * Created by lucieburgess on 27/08/2017.
  * This class is a Transformer according to the Spark Pipeline model
  * Note for each Estimator there is a companion class, Estimator model, which applies the Estimator to a dataset
  *
  */
class ELMModel(override val uid: String, val coefficients: Vector)
  extends ClassificationModel[Vector, ELMModel]
    with ELMParamsMine with DefaultParamsWritable {

  // This uses the default implementation of transform(), which reads column "features" and outputs
  // columns "prediction" and "rawPrediction."

  // This uses the default implementation of predict(), which chooses the label corresponding to
  // the maximum value returned by [[predictRaw()]], as stated in the Classifier Model API

 //FIXME - do we actually need to override def rawPrediction? - see Holden Karau

  override def copy(extra: ParamMap): ELMModel = {
    val copied = new ELMModel(uid, coefficients)
    copyValues(copied, extra).setParent(parent)
  }

  /** Number of classes the label can take. 2 indicates binary classification */
  override val numClasses: Int = 2

  /** Number of features the model was trained on */
  override val numFeatures: Int = coefficients.size

  override def transformSchema(schema: StructType): StructType = schema

  override def transform(dataset: Dataset[_]): DataFrame = super.transform(dataset)

  //FIXME - this won't compile as there is no access to BLAS
  // NB. Implements method predictRaw(features: FeaturesType): Vector
  // Simply matrix multiplication of the features column (a vector of length
  // features is a Vector of
  override def predictRaw(features: Vector): Vector = {
    val margin = BLAS.dot(features, coefficients) // this is just a matrix multipication but no access to the BLAS library
    Vectors.dense(-margin, margin) //
  }

  /** Raw prediction for every possible label */
  
  override def predictRaw(features: Vector): Vector = {

    //coefficients is a Vector of length numFeatures: val coefficients = Vectors.zeros(numFeatures)
    val coefficientsArray = coefficients.toArray
    val coefficientsMatrix: SparkDenseMatrix = new SparkDenseMatrix(numFeatures, 1, coefficientsArray)
    val margin: Array[Double] = coefficientsMatrix.multiply(features).toArray // contains a single element
    val rawPredictions: Array[Double] = Array(-margin(0),margin(0))
    new SparkDenseVector(rawPredictions)
  }
}