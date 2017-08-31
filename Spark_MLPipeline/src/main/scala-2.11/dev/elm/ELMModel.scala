package dev.elm

import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.ml.linalg.{Vector, DenseVector=>SparkDenseVector, DenseMatrix => SparkDenseMatrix}
import org.apache.spark.ml.param.{ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * Created by lucieburgess on 27/08/2017.
  * This class is a Transformer according to the Spark Pipeline model
  * Note for each Estimator there is a companion class, Estimator model, which applies the Estimator to a dataset
  *
  * This uses the default implementation of transform(), which reads column "features" and
  * outputs columns "prediction" and "rawPrediction" based on the ELMClassifier trained using column "features".
  *
  * This uses the default implementation of predict(), which chooses the label corresponding to
  * the maximum value returned by [[predictRaw()]], as stated in the Classifier Model API
  *
  * It also uses the default implementation of
  */
class ELMModel(override val uid: String, val coefficients: Vector)
  extends ClassificationModel[Vector, ELMModel]
    with ELMParams with DefaultParamsWritable {

  override def copy(extra: ParamMap): ELMModel = {
    val copied = new ELMModel(uid, coefficients)
    copyValues(copied, extra).setParent(parent)
  }

  /** Number of classes the label can take. 2 indicates binary classification */
  override val numClasses: Int = 2

  /** Number of features the model was trained on */
  override val numFeatures: Int = coefficients.size

//  override def transformSchema(schema: StructType): StructType = super.transformSchema(schema)

  //override def transform(dataset: Dataset[_]): DataFrame = super.transform(dataset)

  /**
    * Raw prediction for every possible label. Fairly simple implementation based on dot product of (coefficients, features)
    * NB. BLAS not available so had to convert structures to arrays then to matrices and use Spark DenseMatrix class
    * to do matrix mulitplication
    */
  override def predictRaw(features: Vector): Vector = {

    //coefficients is a Vector of length numFeatures: val coefficients = Vectors.zeros(numFeatures)
    val coefficientsArray = coefficients.toArray
    // This matrix is transposed so that the matrix multiplication works. Needs to have (numFeatures) columns and 1 row.
    // Otherwise we get: java.lang.IllegalArgumentException: requirement failed: The columns of A don't match the number of elements of x. A: 1, x: 6
    val coefficientsMatrix: SparkDenseMatrix = new SparkDenseMatrix(1, numFeatures, coefficientsArray)
    val margin: Array[Double] = coefficientsMatrix.multiply(features).toArray // contains a single element
    val rawPredictions: Array[Double] = Array(-margin(0),margin(0))
    new SparkDenseVector(rawPredictions)
  }
}