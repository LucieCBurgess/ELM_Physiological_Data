package elm_test

import breeze.linalg.{*, pinv, DenseMatrix => BDM, DenseVector => BDV}
import breeze.linalg._
import breeze.numerics._
import breeze.numerics.sigmoid
import dev.data_load.DataLoad
import org.apache.spark.ml.linalg.{Vector, DenseVector => SDV, DenseMatrix => SDM}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorSlicer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, monotonically_increasing_id, when}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.scalatest.{BeforeAndAfter, FunSuite}


/**
  * Created by lucieburgess on 04/09/2017.
  */
class ELMPredictRawTest extends FunSuite with BeforeAndAfter {

  lazy val spark: SparkSession = {
    SparkSession.builder().master("local[*]").appName("ELMModelPredictRaw_testing").getOrCreate()
  }

  import spark.implicits._
  val fileName: String = "smalltest.txt" //Has 22 rows of data with 3 features

  /** Load training and test data and cache it */
  val data = DataLoad.createDataFrame(fileName) match {
    case Some(df) => df
      .filter($"activityLabel" > 0)
      .withColumn("binaryLabel", when($"activityLabel".between(1, 3), 0).otherwise(1))
      .withColumn("uniqueID", monotonically_increasing_id())
    case None => throw new UnsupportedOperationException("Couldn't create DataFrame")
  }

  val N: Int = data.count().toInt

  val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z")
  val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
  val dataWithFeatures: DataFrame = featureAssembler.transform(data)
  val numFeatures = featureCols.length

  def extractFeaturesMatrix(ds: Dataset[_]): BDM[Double] = {
    val array = ds.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
    new BDM[Double](numFeatures, N, array)
  }

  def extractLabelsMatrix(ds: Dataset[_]): BDV[Double] = {
    val array = ds.select("binaryLabel").as[Double].collect()
    new BDV[Double](array)
  }

  val L = 10
  val bias: BDV[Double] = BDV.rand[Double](L) // L x 1
  val weights: BDM[Double] = BDM.rand[Double](L, numFeatures) // L x numFeatures
  val X: BDM[Double] = extractFeaturesMatrix(dataWithFeatures) // numFeatures x N
  val T: BDV[Double] = extractLabelsMatrix(dataWithFeatures)

  val M = weights * X //L x N

  val H: BDM[Double] = sigmoid((M(::,*) + bias).t) // We want H to be N x L so that pinv(H) is L x N

  val beta: BDV[Double] = pinv(H) * T // (L x N) . N => gives vector of length L

  /** Try again simple simple matrix multiplication: T = Beta x H where H is calculated from the new dataset */
  test("[02] Can return the labels in one go using T = Beta.H(transpose)") {

    assert(beta.length == 10)

    def predictRaw(features: DataFrame) :SDV = {

      //val features: RDD[Vector] = dataWithFeatures.select("features").rdd.map(r => r.getAs[Vector](0))

      val featuresArray: Array[Double] = features.rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
      val featuresMatrix = new BDM[Double](numFeatures, N, featuresArray)

      val M = weights * featuresMatrix // L x numFeatures. numFeatures x N where N is no. of test samples. NB Features must be of size (numFeatures, N)
      val H = sigmoid((M(::, *)) + bias) // L x numFeatures
      val T = beta.t * H // L.(L x N) of type Transpose[DenseVector]
      new SDV((T.t).toArray) //length N
    }

    val features = dataWithFeatures.select("features")
    val predictions: SDV = predictRaw(features)
    assert(predictions.isInstanceOf[SDV])
    assert(predictions.size == N)
    println(predictions.values.mkString(","))

  }

  /** This test is failing because the operation over the RDD is not serialisable. So, back to arrays again */
  test("[01]. Can return a label for a single sample") {

    val features: RDD[Vector] = dataWithFeatures.select("features").rdd.map(r => r.getAs[Vector](0))

    def predictRaw(features: RDD[Vector]) :SDV = {
      val outputArray: Array[Double] = features.map(f => calculateNode(f)).collect() //Exception: calculateNode(f) not serializable
      new SDV(outputArray)
    }

    def calculateNode(feature: Vector): Double = {
      val featureBDV = new BDV[Double](feature.toArray)
      val node: IndexedSeq[Double] = for (j <- 0 until L) yield (beta(j) * sigmoid(weights(j, ::) * featureBDV + bias(j)))
      node.sum.round.toDouble
    }

    val predictions: SDV = predictRaw(features)
    assert(predictions.isInstanceOf[SDV])
    assert(predictions.size == N)
  }
}
