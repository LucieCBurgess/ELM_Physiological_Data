package elm_test

import breeze.linalg.{*, pinv, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import dev.data_load.DataLoadOption
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{monotonically_increasing_id, when}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.scalatest.FunSuite

/**
  * Created by lucieburgess on 02/09/2017.
  * ALL TESTS PASS 11/09/2017 but spark.stop() has to be removed from the end of the file
  */
class ELMAlgoTest extends FunSuite {

  val spark: SparkSession = {
    SparkSession.builder().master("local[*]").appName("elm_algorithm testing").getOrCreate()
  }

  import spark.implicits._

  val smallFile: String = "smalltest.txt" //smallFile has 22 rows of data with 3 features
  val bigFile: String = "mHealth_subject1.txt" //bigFile has 35,174 rows with 3 features

  /** Small file: Load training and test data and cache it */
  val smallData: DataFrame = DataLoadOption.createDataFrame(smallFile) match {
    case Some(df) => df
      .filter($"activityLabel" > 0)
      .withColumn("binaryLabel", when($"activityLabel".between(1, 3), 0).otherwise(1))
      .withColumn("uniqueID", monotonically_increasing_id())
    case None => throw new UnsupportedOperationException("Couldn't create DataFrame")
  }

  /** Big file: Load training and test data and cache it */
  val bigData: DataFrame = DataLoadOption.createDataFrame(bigFile) match {
    case Some(df) => df
      .filter($"activityLabel" > 0)
      .withColumn("binaryLabel", when($"activityLabel".between(1, 3), 0).otherwise(1))
      .withColumn("uniqueID", monotonically_increasing_id())
    case None => throw new UnsupportedOperationException("Couldn't create DataFrame")
  }

  val smallN: Int = smallData.count().toInt
  val bigN: Int = bigData.count().toInt

  val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z")
  val featureAssembler: VectorAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
  val numFeatures: Int = featureCols.length

  val smallDF: DataFrame = featureAssembler.transform(smallData)
  val bigDF: DataFrame = featureAssembler.transform(bigData)


  /** Test for construction of the labels vector */
  test("[01] Can create an array of doubles from a single DataFrame column") {

    val array: Array[Double] = smallDF.select("acc_Chest_X").as[Double].collect()
    assertResult(array.length) {
      smallDF.count().toInt
    }

    val array2: Array[Double] = bigDF.select("acc_Chest_X").as[Double].collect()
    assertResult(array2.length) {
      bigDF.count().toInt
    }
  }

  /** Test for construction of the features matrix */
  test("[02] Can create an array of all the data in a features Vector") {

    val array = smallDF.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
    assertResult(smallN * numFeatures) {
      array.length
    }

    val array2 = bigDF.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
    assertResult(bigN * numFeatures) {
      array2.length
    }
  }

  /** Test for construction of the features matrix - checks that the BDM is of the correct size
    * This is needed because extracting the feature vector into an array from an RDD and into a BDM
    * of column-major order effectively transposes the matrix
    * In other words, we have to transpose the number of rows and columns of the features vector
    * and then transpose the resulting matrix, to get the matrix we want
    */
  test("[03] Computing X (features matrix) results in a matrix of correct cardinality") {

    val array = smallDF.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
    val simpleArray = Array(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0)//3 feature vectors with 5 samples
    val parallelData: RDD[Double] = spark.sparkContext.parallelize(simpleArray)
    val X = new BDM[Double](5,3, simpleArray) // gives column major order so 5 rows and 3 columns
    println(X.data.mkString(","))
    assert(X(0,0) === 1.0)
    assert(X(1,0) === 2.0)
    assert(X(2,0) === 3.0)
    assert(X(3,0) === 4.0)
    assert(X(4,0) === 5.0)
    assert(X(0,1) === 6.0)
    assert(X(1,1) === 7.0)
    assert(X(2,1) === 8.0)
    assert(X(3,1) === 9.0)
    assert(X(4,1) === 10.0)
    assert(X(0,2) === 11.0)
    assert(X(1,2) === 12.0)
    assert(X(2,2) === 13.0)
    assert(X(3,2) === 14.0)
    assert(X(4,2) === 15.0)

    val featuresArray = Array(1.0,2.0,3.0,1.0,2.0,3.0,1.0,2.0,3.0,1.0,2.0,3.0,1.0,2.0,3.0)
    val F = new BDM[Double](3,5, featuresArray) //NB. 3 rows and 5 columns because we will transpose to get features in right place
    println(F.data.mkString(","))
    assert(F(0,0) === 1.0)
    assert(F(1,0) === 2.0)
    assert(F(2,0) === 3.0)
    assert(F(0,1) === 1.0)
    assert(F(1,1) === 2.0)
    assert(F(2,1) === 3.0)
    assert(F(0,2) === 1.0)
    assert(F(1,2) === 2.0)
    assert(F(2,2) === 3.0)
    assert(F(0,3) === 1.0)
    assert(F(1,3) === 2.0)
    assert(F(2,3) === 3.0)
    assert(F(0,4) === 1.0)
    assert(F(1,4) === 2.0)
    assert(F(2,4) === 3.0)

    val Z = F.t
    println(Z.data.mkString(","))
    assert(Z(0,0) === 1.0)
    assert(Z(1,0) === 1.0)
    assert(Z(2,0) === 1.0)
    assert(Z(3,0) === 1.0)
    assert(Z(4,0) === 1.0)
    assert(Z(0,1) === 2.0)
    assert(Z(1,1) === 2.0)
    assert(Z(2,1) === 2.0)
    assert(Z(3,1) === 2.0)
    assert(Z(4,1) === 2.0)
    assert(Z(0,2) === 3.0)
    assert(Z(1,2) === 3.0)
    assert(Z(2,2) === 3.0)
    assert(Z(3,2) === 3.0)
    assert(Z(4,2) === 3.0)
  }

  test("[04] Check size of the dataset is 35,174 lines for mHealth_subject1.txt and 22 lines for smalltest") {
    assertResult(22) { smallN }
    assertResult(35174) { bigN }
  }

  test("[05] Can compute Bias as a Breeze Dense Vector of length L") {

    val L = 10
    val bias: BDV[Double] = BDV.rand[Double](L)
    assert(bias.length == 10)
    println(bias.data.mkString(","))
    assert(bias.isInstanceOf[BDV[Double]])
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

    val smallFeatures: Int = smallDF.select("features").head.get(0).asInstanceOf[Vector].size
    assert(smallFeatures === numFeatures)

    val bigFeatures: Int = bigDF.select("features").head.get(0).asInstanceOf[Vector].size
    assert(bigFeatures === numFeatures)

    /** NB. Do not use this function in the model, purely for testing. Use extractFeaturesMatrix instead */
    def extractSmallFeaturesMatrixTranspose(ds: Dataset[_]): BDM[Double] = {
      val array = ds.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
      new BDM[Double](numFeatures, smallN, array).t //NB X gets transposed again to check values in smalltest
    }

    val X: BDM[Double] = extractSmallFeaturesMatrixTranspose(smallDF)
    assert(X.rows == 22) // number of samples
    assert(X.cols == 3) // number of features
    assert(X(0,0) == -9.5357)
    assert(X(7,1) == -0.017569)
    assert(X(17,2) == 0.92812)
    assert(X.isInstanceOf[BDM[Double]])

    /** NB. Do not use this function in the model, purely for testing. Use extractFeaturesMatrix instead */
    def extractBigFeaturesMatrixTranspose(ds: Dataset[_]): BDM[Double] = {
      val array = ds.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
      new BDM[Double](numFeatures, bigN, array).t //NB X gets transposed again to check values in smalltest
    }

    val Y: BDM[Double] = extractBigFeaturesMatrixTranspose(bigDF)
    assert(Y.rows == 35174) // number of samples
    assert(Y.cols == 3) // number of features
    assert(Y.isInstanceOf[BDM[Double]])
  }

  /** Only try this for smalltest. It fails for large files due to pinv
    * Not used in the model as Breeze allows column forecasting. This method works by calculating bias as a
    * matrix instead of a vector - useful method for when using Spark matrices as they do not have the
    * broadcast functionality */
  test("[08] Can calculate H from weights, bias and X (features) using bias matrix and ListBuffer") {

    def extractFeaturesMatrix(ds: Dataset[_]): BDM[Double] = {
      val array = ds.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
      new BDM[Double](numFeatures, smallN, array)
    }

    def extractLabelsVector(ds: Dataset[_]): BDV[Double] = {
      val array = ds.select("binaryLabel").as[Double].collect()
      new BDV[Double](array)
    }

    val L = 10
    val bias: BDM[Double] = BDM.rand[Double](L,1) // L x 1
    val weights: BDM[Double] = BDM.rand[Double](L, numFeatures) // L x numFeatures
    val X: BDM[Double] = extractFeaturesMatrix(smallDF) // numFeatures x N
    assert(X.rows == 3)
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

    //No longer needed when using Breeze matrices, can use the broadcast function
    val biasArray = bias.toArray // bias of Length L
    val buf = scala.collection.mutable.ArrayBuffer.empty[Array[Double]]
      for (i <- 0 until smallN) yield buf += biasArray
    val replicatedBiasArray = buf.flatten.toArray

    // bias is N x L, but the same column vector (length L) repeated N times
    val bias2 :BDM[Double] = new BDM[Double](smallN,L,replicatedBiasArray)

    val H = sigmoid(M + bias2) // We want H to be N x L

    assert(H.isInstanceOf[BDM[Double]])
    assert(H.rows == 22)
    assert(H.cols == 10)

    val T: BDV[Double] = extractLabelsVector(smallDF)

    val beta: BDV[Double] = pinv(H) * T

    assert(beta.isInstanceOf[BDV[Double]])
    assert(beta.length == 10)

  }

  test("[09] Can calculate H from weights, bias and X (features) using bias vector and Breeze column broadcasting") {

    def extractFeaturesMatrix(ds: Dataset[_]): BDM[Double] = {
      val array = ds.select("features").rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
      new BDM[Double](numFeatures, smallN, array)
    }

    def extractLabelsVector(ds: Dataset[_]): BDV[Double] = {
      val array = ds.select("binaryLabel").as[Double].collect()
      new BDV[Double](array)
    }

    val L = 10
    val bias: BDV[Double] = BDV.rand[Double](L) // L x 1
    val weights: BDM[Double] = BDM.rand[Double](L, numFeatures) // L x numFeatures
    val X: BDM[Double] = extractFeaturesMatrix(smallDF) // numFeatures x N
    assert(X.rows == 3)
    assert(X.cols == 22)
    assert(X.isInstanceOf[BDM[Double]])

    val M = weights * X //LXF.FxN = LxN

    assert(M.isInstanceOf[BDM[Double]])
    assert(M.rows == 10) // L x N
    assert(M.cols == 22)

    val Z: BDM[Double] = (M(::,*) + bias).t // We want H to be N x L so that pinv(H) is L x N

    def calculateH(af: String, Z: BDM[Double]): BDM[Double] = af match {
      case "sigmoid" => sigmoid(Z)
      case "tanh" => tanh(Z)
      case "sin" => sin(Z)
      case _ => throw new IllegalArgumentException("Activation function must be sigmoid, tanh, or sin")
    }

    val H1 = calculateH("sigmoid", Z)
    val H2 = calculateH("tanh", Z)
    val H3 = calculateH("sin", Z)

    //val H: BDM[Double] = sigmoid((M(::,*) + bias).t) // We want H to be N x L so that pinv(H) is L x N

    assert(H1.isInstanceOf[BDM[Double]]) // N x L
    assert(H1.rows == 22)
    assert(H1.cols == 10)

    assert(H2.isInstanceOf[BDM[Double]]) // N x L
    assert(H2.rows == 22)
    assert(H2.cols == 10)

    assert(H3.isInstanceOf[BDM[Double]]) // N x L
    assert(H3.rows == 22)
    assert(H3.cols == 10)

    intercept[IllegalArgumentException] {
      val H4 = calculateH("custard", Z)
    }

    val T: BDV[Double] = extractLabelsVector(smallDF)

    val beta: BDV[Double] = pinv(H1) * T // Length L because (L x N) * N gives vector of length L

    assert(beta.isInstanceOf[BDV[Double]])
    assert(beta.length == 10)

  }
}


