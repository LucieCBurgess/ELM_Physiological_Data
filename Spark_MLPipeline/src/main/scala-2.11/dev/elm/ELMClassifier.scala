package dev.elm

import dev.data_load.SparkSessionWrapper
import org.apache.spark.ml.classification.Classifier
import org.apache.spark.ml.linalg.{Vector, DenseVector => SDV, DenseMatrix => SDM}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset}
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}

/**
  * Created by lucieburgess on 27/08/2017.
  * This is a concrete Estimator (Classifier) of type ELMEstimator. Conforms to the following API:
  * https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.classification.Classifier
  * Has access to methods in ClassificationModel through extending ELMModel. Not all of these are public according to the
  * documentation yet seem to be available e.g. getNumClasses
  * Clean version which uses the DeveloperAPI example
  */
class ELMClassifier(val uid: String) extends Classifier[Vector, ELMClassifier, ELMModel]
  with ELMParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("ELM Estimator algorithm which includes train() method"))

  override def copy(extra: ParamMap): ELMClassifier = defaultCopy(extra)

  /** Set parameters */
  def setActivationFunc(value: String): this.type = set(activationFunc, value)

  def setHiddenNodes(value: Int): this.type = set(hiddenNodes, value)

  def setFracTest(value: Double): this.type = set(fracTest, value)

  /**
    * Implements method in org.apache.spark.ml.Predictor. This method is used by fit(). Uses the default version of transformSchema
    * Trains the model to predict the training labels based on the ELMAlgorithm class.
    * @param ds the dataset to be operated on
    * @return an ELMModel which extends ClassificationModel with the output weight vector beta calculated, of length L,
    *         where L is the number of hidden nodes.
    */
  override def train(ds: Dataset[_]): ELMModel = {

    import ds.sparkSession.implicits._ // was import ds.sparkSession.implicits._ ... does it matter?
    //transformSchema(ds.schema, logging = true) // if you don't include this line you don't get a features column
    ds.cache()
    ds.printSchema()

    val numClasses = getNumClasses(ds)
    println(s"This is a binomial classifier and the number of class should be 2: it is $numClasses")


    val modelHiddenNodes: Int = getHiddenNodes
    val af: String = getActivationFunc

    val eLMClassifierAlgo = new ELMClassifierAlgo(ds, modelHiddenNodes, af) // Changed from ELMClassifierAlgo
    val modelBias: BDM[Double] = eLMClassifierAlgo.bias
    val modelWeights: BDM[Double] = eLMClassifierAlgo.weights
    val modelBeta: BDV[Double] = eLMClassifierAlgo.calculateBeta()
    val modelAF: ActivationFunction = eLMClassifierAlgo.chosenAF // has to be of type ActivationFunc not String

    // beta is effectively the coefficients and then we write transform in ELMModel,
    // or alternatively in ELMClassifierAlgo and pass it back to ELMModel. The transform is essentially the prediction.

    val model = new ELMModel(uid, modelBias, modelWeights, modelBeta, modelHiddenNodes, modelAF).setParent(this)
    model
  }
}

/** Companion object enables deserialisation of ELMParamsMine */
object ELMClassifier extends DefaultParamsReadable[ELMClassifier]


/** This is the previous version using a blank coefficients vector which runs OK but doesn't actually learn anything */
//FIXME - logic of the model would happen here.
// Do learning to estimate the coefficients vector.
//val coefficients = Vectors.zeros(numFeatures)
// val model = new ELMModel (uid, coefficients).setParent(this)
// model
// Create a model, and return it.
/** This is the previous version using a blank coefficients vector which runs OK but doesn't actually learn anything */
//FIXME - logic of the model would happen here.
// Do learning to estimate the coefficients vector.
//val coefficients = Vectors.zeros(numFeatures)
// val model = new ELMModel (uid, coefficients).setParent(this)
// model
// copyValues(model)
//FIXME - pass one of these to ELMClassifierAlgo
//val featuresVector: SDV = ds.select("features1", "features2", "features3").asInstanceOf[SDV]
//val featuresColVector: DataFrame = ds.select(col($(featuresCol))) //pass this to ELMClassifierAlgo
// val numRows = ds.count()