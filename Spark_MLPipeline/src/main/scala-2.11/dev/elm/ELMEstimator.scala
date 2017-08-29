package dev.elm

import dev.data_load.SparkSessionWrapper
import org.apache.spark.ml.classification.Classifier
import org.apache.spark.ml.linalg.{Vector, Vectors, Matrix, Matrices}
import org.apache.spark.ml.param.{IntParam, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * Created by lucieburgess on 27/08/2017.
  * Has access to methods in ClassificationModel through extending ELMModel e.g.
  */
class ELMEstimator(val uid: String) extends Classifier[Vector, ELMEstimator, ELMModel]
  with ELMParamsMine with DefaultParamsWritable with SparkSessionWrapper {

  def this() = this(Identifiable.randomUID("ELM Estimator algorithm which includes fit() method"))

  //FIXME - use MinMaxScaler to set the RawPredictions column to a value of (0,1) which helps give final labels

  override def copy(extra: ParamMap): ELMEstimator = defaultCopy(extra)

  /** Set parameters */

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setActivationFunc(value: String): this.type = set(activationFunc, value) // has to be in this class since returns type ELMEstimator

  setHiddenNodes(10)
  def setHiddenNodes(value: Int): this.type = set(hiddenNodes, value)

  def setFracTest(value: Double): this.type = set(fracTest, value)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  //Implements method in Predictor. Not sure why to use train() rather than fit()
  // FIXME - this contains the logic of the model - to write
  // FIXME - where do we take the feature columns and map these into a vector, like in the Logistic Regression example?
  // FIXME - where do we set activityLabel as labelCol?
  override def train(ds: Dataset[_]): ELMModel = {

    import ds.sparkSession.implicits._

    transformSchema(ds.schema, logging = true)
    ds.cache()
    val datasetSize = ds.count() //gives the total number of training or (train, test) examples

    val numClasses = getNumClasses(ds) // states whether this is a binomial or a multinomial classifier, should be 2

    // Get the number of features by peeking at the first row in the dataset
    // FIXME bit confused about these because we should have the ability to set numFeatures
    val numFeatures: Int = ds.select(col($(featuresCol))).head.get(0).asInstanceOf[Vector].size

    // Determine the number of records for each class[0 or 1]
    val groupedByLabel = ds.select(col($(labelCol)).as[Double]).groupByKey(x => x)

    // Do learning to estimate the coefficients vector.
    //FIXME - logic of the model would happen here.
    // val coefficients = Vectors.zeros(numFeatures)
    val coefficients = Vectors.zeros(numFeatures)

    // Unpersist the dataset now that we have trained it.
    ds.unpersist()

    // Create a model, and return it.
    val model = new ELMModel(uid, numFeatures, coefficients).setParent(this)

    copyValues(model)
  }
}

// Companion object enables deserialisation of ELMParamsMine
object ELMEstimator extends DefaultParamsReadable[ELMEstimator]
