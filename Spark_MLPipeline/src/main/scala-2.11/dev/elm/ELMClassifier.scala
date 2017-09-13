package dev.elm

import org.apache.spark.ml.classification.Classifier
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.Dataset
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}

/**
  * Created by lucieburgess on 27/08/2017.
  * This is a concrete Estimator (Classifier) of type ELMEstimator. Conforms to the following API:
  * https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.classification.Classifier
  * Has access to methods in ClassificationModel through extending ELMModel.
  */
class ELMClassifier(val uid: String) extends Classifier[Vector, ELMClassifier, ELMModel]
  with ELMParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("ELM Estimator algorithm"))

  override def copy(extra: ParamMap): ELMClassifier = defaultCopy(extra)

  /** Set additional parameters; activationFunc, hiddenNodes, fracTest */
  def setActivationFunc(value: String): this.type = set(activationFunc, value)

  def setHiddenNodes(value: Int): this.type = set(hiddenNodes, value)

  def setFracTest(value: Double): this.type = set(fracTest, value)

  /**
    * Implements method in org.apache.spark.ml.Predictor. This method is used by fit().
    * Trains the model to predict the training labels based on the ELMAlgorithm class.
    * Uses the default version of transformSchema
    * @param ds the dataset to be operated on
    * @return an ELMModel which extends ClassificationModel with the output weight vector modelBeta calculated, of length L,
    *         where L is the number of hidden nodes, with modelBias vector and modelWeights matrix also calculated.
    *         Driven by algorithm in ELMClassifierAlgo.
    */
  override def train(ds: Dataset[_]): ELMModel = {

    ds.cache()
    ds.printSchema()

    val numClasses: Int = getNumClasses(ds)
    println(s"This is a binomial classifier and the number of classes is $numClasses")

    val modelHiddenNodes: Int = getHiddenNodes
    val modelAF: String = getActivationFunc

    val elmAlgo = new ELMAlgo(ds, modelHiddenNodes, modelAF)
    val modelWeights: BDM[Double] = elmAlgo.weights
    val modelBias: BDV[Double] = elmAlgo.bias
    val modelBeta: BDV[Double] = elmAlgo.calculateBeta()
    val modelNumFeatures: Int = elmAlgo.algoNumFeatures

    /** Return the trained model */
    val model = new ELMModel(uid, modelWeights, modelBias, modelBeta, modelHiddenNodes, modelAF, modelNumFeatures).setParent(this)
    model
  }
}

/** Companion object enables deserialisation of ELMParams */
object ELMClassifier extends DefaultParamsReadable[ELMClassifier]
