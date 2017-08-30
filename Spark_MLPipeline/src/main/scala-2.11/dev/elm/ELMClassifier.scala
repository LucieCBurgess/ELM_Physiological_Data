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
  * This is a concrete Estimator (Classifier) of type ELMEstimator. Conforms to the following API:
  * https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.classification.Classifier
  * Has access to methods in ClassificationModel through extending ELMModel. Not all of these are public according to the
  * documentation yet seem to be available e.g. getNumClasses
  * Clean version which uses the DeveloperAPI example
  */
class ELMClassifier(val uid: String) extends Classifier[Vector, ELMClassifier, ELMModel]
  with ELMParams with DefaultParamsWritable with SparkSessionWrapper {

  def this() = this(Identifiable.randomUID("ELM Estimator algorithm which includes fit() method"))

  override def copy(extra: ParamMap): ELMClassifier = defaultCopy(extra)

  /** Set parameters */
  def setActivationFunc(value: String): this.type = set(activationFunc, value)

  def setHiddenNodes(value: Int): this.type = set(hiddenNodes, value)

  def setFracTest(value: Double): this.type = set(fracTest, value)


  //Implements method in Predictor. This method is used by fit()
  // According to Predictor, this is the method that developers need to implement and can avoid dealing with schema validation
  // and copying parameters into the model
  override def train(ds: Dataset[_]): ELMModel = {

    import ds.sparkSession.implicits._

    //transformSchema(ds.schema, logging = true) // if you don't include this line you don't get a features column
    ds.cache()
    println("**************** printing the training dataset schema in the TRAIN() function within ELMClassifier ********************")
    ds.printSchema()
    val datasetSize = ds.count() //gives the total number of training or (train, test) examples

    val numClasses = getNumClasses(ds) // states whether this is a binomial or a multinomial classifier, should be 2

    // Get the number of features by peeking at the first row in the dataset
    val numFeatures: Int = ds.select(col($(featuresCol))).head.get(0).asInstanceOf[Vector].size

    // Determine the number of records for each class[0 or 1]
    val groupedByLabel = ds.select(col($(labelCol)).as[Double]).groupByKey(x => x)

    // Do learning to estimate the coefficients vector.
    //FIXME - logic of the model would happen here.
    //FIXME - I guess we would use FracTest here as the model is trained on only $FracTest% of the data.
    val coefficients = Vectors.zeros(numFeatures)

    // Unpersist the dataset now that we have trained it.
    //ds.unpersist()

    // Create a model, and return it.
    val model = new ELMModel(uid, coefficients).setParent(this)
    model

    //copyValues(model)
  }
}

// Companion object enables deserialisation of ELMParamsMine
object ELMClassifier extends DefaultParamsReadable[ELMClassifier]
