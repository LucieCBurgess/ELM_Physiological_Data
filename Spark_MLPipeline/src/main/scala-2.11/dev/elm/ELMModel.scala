package dev.elm

import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.ml.linalg.{Vector, DenseMatrix => SDM, DenseVector => SDV}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, Vector => BV}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * Created by lucieburgess on 27/08/2017. LOTS OF WORK to do, this does not compile ...
  * This class is a Transformer according to the Spark Pipeline model
  * Note for each Estimator there is a companion class, Estimator model, which applies the Estimator to a dataset
  *
  * This uses the default implementation of transform(), which reads column "features" and
  * outputs columns "prediction" and "rawPrediction" based on the ELMClassifier trained using column "features".
  *
  * This uses the default implementation of predict(), which chooses the label corresponding to
  * the maximum value returned by [[predictRaw()]], as stated in the Classifier Model API
  *
  * It also uses the default implementation of transform(ds: DataSet) and transformSchema( schema: StructType) in the Predicotr model.
  * Note these methods are not passing correctly to the Pipeline API due to some bug in the Spark source code.
  * As a result the features vector is not passing correctly through from ELMClassifier to ELMModel
  * Therefore I have added the features vector manually to the DataFrame used in the pipeline.
  * // was class ELMModel (override val uid: String, val coefficients: Vector)
  */
class ELMModel(override val uid: String, val modelBeta: BDV[Double])
  extends ClassificationModel[Vector, ELMModel]
    with ELMParams with DefaultParamsWritable {

  override def copy(extra: ParamMap): ELMModel = {
    val copied = new ELMModel(uid, modelBeta)
    copyValues(copied, extra).setParent(parent)
  }

  /** Number of classes the label can take. 2 indicates binary classification */
  override val numClasses: Int = 2

  /** Number of features the model was trained on */
  //FIXME this is hard-coded for now ...
  override val numFeatures: Int = 6
  // Get the number of features by peeking at the first row in the dataset
  //override val numFeatures: Int = ds.select(col($(featuresCol))).head.get(0).asInstanceOf[Vector].size

  override def transformSchema(schema: StructType): StructType = super.transformSchema(schema)

  /** Takes a dataframe, uses beta and ouputs a vector of classification labels */
//  override def transform(ds: Dataset[_]): DataFrame = {
//    transformSchema(ds.schema, logging = true)
//
//
//    val eLMClassifierAlgo = new ELMClassifierAlgo(ds, labels, af: activationFunc = getActivationFunc)
//    this.getPredictionCol = eLMClassifierAlgo.predictAllLabels(ds)
//
//  }

  def predictRaw(features: SDV) :Vector = {
    ???
  }

  /**
    * Raw prediction for every possible label. Fairly simple implementation based on dot product of (coefficients, features)
    * NB. BLAS not available so had to convert structures to arrays then to matrices and use Spark DenseMatrix class
    * to do matrix mulitplication
    * Need to update this function as originally predictRaw was based on coefficients, which is a vector of numFeatures
    * Our predictRaw needs to be based on beta so I'm not sure how to calculate it
    */
//  override def predictRaw(features: Vector): Vector = {
//
//    //coefficients is a Vector of length numFeatures: val coefficients = Vectors.zeros(numFeatures)
//    val coefficientsArray = coefficients.toArray
//    // This matrix is transposed so that the matrix multiplication works. Needs to have (numFeatures) columns and 1 row.
//    // Otherwise we get: java.lang.IllegalArgumentException: requirement failed: The columns of A don't match the number of elements of x. A: 1, x: 6
//    val coefficientsMatrix: SparkDenseMatrix = new SparkDenseMatrix(1, numFeatures, coefficientsArray)
//    val margin: Array[Double] = coefficientsMatrix.multiply(features).toArray // contains a single element
//    val rawPredictions: Array[Double] = Array(-margin(0),margin(0))
//    new SparkDenseVector(rawPredictions)
//  }
}