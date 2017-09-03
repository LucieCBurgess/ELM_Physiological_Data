package elm_test

import breeze.linalg.{pinv, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.sigmoid
import dev.data_load.DataLoad
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorSlicer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, monotonically_increasing_id, when}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.scalatest.FunSuite

/**
  * Created by lucieburgess on 02/09/2017.
  */
class ELMAlgoTest extends FunSuite {

  lazy val spark: SparkSession = {
    SparkSession.builder().master("local[*]").appName("ELMAlgoTest").getOrCreate()
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

  def extractFeaturesMatrixTranspose(ds: Dataset[_]): BDM[Double] = {
    val array = ds.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
    new BDM[Double](numFeatures, N, array).t //NB X gets transposed again above, so no need to transpose, but will do this for t
  }

  def extractFeaturesMatrix(ds: Dataset[_]): BDM[Double] = {
    val array = ds.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
    new BDM[Double](numFeatures, N, array)
  }

  def extractLabelsMatrix(ds: Dataset[_]): BDV[Double] = {
    val array = ds.select("binaryLabel").as[Double].collect()
    new BDV[Double](array)
  }

  test("[01] Can create an array of doubles from a single DataFrame column") {

    val array: Array[Double] = dataWithFeatures.select("acc_Chest_X").as[Double].collect()
    assertResult(array.length) {
      dataWithFeatures.count().toInt
    }
  }

  test("[02] Can create an array of all the data in a features Vector") {

    val array = dataWithFeatures.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
    assertResult(N * 3) {
      array.length
    }
  }

  test("[03] Computing X (features matrix) results in a matrix of correct cardinality") {

    val array = dataWithFeatures.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
    val simpleArray = Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0)//3 feature vectors with 5 samples
    val parallelData: RDD[Double] = spark.sparkContext.parallelize(simpleArray)
    val X = new BDM[Double](5,3, simpleArray) // gives column major order so 5 rows and 3 columns
    println(X.data.mkString(","))
    assert(X(0,0) == 1.0)
    assert(X(1,0) == 2.0)
    assert(X(2,0) == 3.0)
    assert(X(3,0) == 4.0)
    assert(X(4,0) == 5.0)
    assert(X(0,1) == 6.0)
    assert(X(1,1) == 7.0)
    assert(X(2,1) == 8.0)
    assert(X(3,1) == 9.0)
    assert(X(4,1) == 10.0)
    assert(X(0,2) == 11.0)
    assert(X(1,2) == 12.0)
    assert(X(2,2) == 13.0)
    assert(X(3,2) == 14.0)
    assert(X(4,2) == 15.0)

    val featuresArray = Array(1.0,2.0,3.0,1.0,2.0,3.0,1.0,2.0,3.0,1.0,2.0,3.0,1.0,2.0,3.0)
    val F = new BDM[Double](3,5, featuresArray) //NB. 3 rows and 5 columns because we will transpose to get features in right place
    println(F.data.mkString(","))
    assert(F(0,0) == 1.0)
    assert(F(1,0) == 2.0)
    assert(F(2,0) == 3.0)
    assert(F(0,1) == 1.0)
    assert(F(1,1) == 2.0)
    assert(F(2,1) == 3.0)
    assert(F(0,2) == 1.0)
    assert(F(1,2) == 2.0)
    assert(F(2,2) == 3.0)
    assert(F(0,3) == 1.0)
    assert(F(1,3) == 2.0)
    assert(F(2,3) == 3.0)
    assert(F(0,4) == 1.0)
    assert(F(1,4) == 2.0)
    assert(F(2,4) == 3.0)

    val Z = F.t
    println(Z.data.mkString(","))
    assert(Z(0,0) == 1.0)
    assert(Z(1,0) == 1.0)
    assert(Z(2,0) == 1.0)
    assert(Z(3,0) == 1.0)
    assert(Z(4,0) == 1.0)
    assert(Z(0,1) == 2.0)
    assert(Z(1,1) == 2.0)
    assert(Z(2,1) == 2.0)
    assert(Z(3,1) == 2.0)
    assert(Z(4,1) == 2.0)
    assert(Z(0,2) == 3.0)
    assert(Z(1,2) == 3.0)
    assert(Z(2,2) == 3.0)
    assert(Z(3,2) == 3.0)
    assert(Z(4,2) == 3.0)

    // In other words, when getting the features vector we have to transpose the number of rows and columns,
    // and then transpose the whole matrix, to get the matrix we want
  }

  test("[04] Check size of the dataset is 35,174 lines for mHealth_subject1.txt and 22 lines for smalltest") {
    assertResult(22) {
      N
    }
  }

  test("[05] Can compute Bias as a Breeze Dense Matrix with L rows and 1 column") {

    val L = 10
    val bias: BDM[Double] = BDM.rand[Double](L,1)
    assert(bias.rows == 10)
    assert(bias.cols == 1)
    println(bias.data.mkString(","))
    assert(bias.isInstanceOf[BDM[Double]])
  }

  test("[06] Can compute Weights as a Breeze Dense Matrix with L rows and numFeatures columns") {
    val L = 10
    val weights: BDM[Double] = BDM.rand[Double](L, numFeatures)
    assert(weights.rows == 10)
    assert(weights.cols == 3)
    println(weights.data.mkString(","))
    assert(weights.isInstanceOf[BDM[Double]])

  }

  /** NB X here is transposed. The real X has numFeatures rows and N columns */
  test("[07] Can create a Breeze Dense Matrix of the data in a features Vector with correct cardinality") {

    val numFeatures: Int = dataWithFeatures.select("features").head.get(0).asInstanceOf[Vector].size

    val X: BDM[Double] = extractFeaturesMatrixTranspose(dataWithFeatures)
    assert(X.rows == N)
    assert(X.cols == numFeatures)
    assert(X(0,0) == -9.5357)
    assert(X(7,1) == -0.017569)
    assert(X(17,2) == 0.92812)
    assert(X.isInstanceOf[BDM[Double]])
  }

  test("[08] Can calculate H from weights, bias and X (features)") {

    val L = 10
    val bias: BDM[Double] = BDM.rand[Double](L,1) // L x 1 SHOULD be L x N with each column being the same
    val weights: BDM[Double] = BDM.rand[Double](L, numFeatures) // L x numFeatures
    val X: BDM[Double] = extractFeaturesMatrix(dataWithFeatures) // numFeatures x N
    assert(X.rows == numFeatures)
    assert(X.cols == 22)
    assert(X.isInstanceOf[BDM[Double]])

    val Z: BDM[Double] = weights * X //L x F . F x N
    assert(Z.isInstanceOf[BDM[Double]])
    assert(Z.rows == 10)
    assert(Z.cols == 22)

    val M = (weights * X).t //N x L

    assert(M.isInstanceOf[BDM[Double]])
    assert(M.rows == 22) // N x L
    assert(M.cols == 10)

    val biasArray = bias.toArray // bias of Length L
    val buf = scala.collection.mutable.ArrayBuffer.empty[Array[Double]]
      for (i <- 0 until N) yield buf += biasArray
    val replicatedBiasArray = buf.flatten.toArray

    val bias2 :BDM[Double] = new BDM[Double](N,L,replicatedBiasArray) // bias is N x L, but the same column vector (length L) repeated N times

    val H = sigmoid(M + bias2) // We want H to be N x L

    assert(H.isInstanceOf[BDM[Double]])
    assert(H.rows == 22)
    assert(H.cols == 10)

    val T: BDV[Double] = extractLabelsMatrix(dataWithFeatures)

    val beta: BDV[Double] = pinv(H) * T

    assert(beta.isInstanceOf[BDV[Double]])
    assert(beta.length == 10)

  }
}
