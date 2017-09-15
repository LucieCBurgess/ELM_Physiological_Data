package elm_test

import breeze.linalg.{*, pinv, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import dev.data_load.DataLoadOption
import org.apache.spark.ml.linalg.{Vector, DenseVector => SDV}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{monotonically_increasing_id, when, udf}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.scalatest.{BeforeAndAfter, FunSuite}

/**
  * Created by lucieburgess on 04/09/2017.
  * Unit tests for methods and predictRaw logic within ELMModel class.
  * ALL TESTS PASS. Test 5 is cancelled, see note below.
  */
class ELMPredictRawTest extends FunSuite with BeforeAndAfter {

  lazy val spark: SparkSession = {
    SparkSession.builder().master("local[*]").appName("ELMModel_testing").getOrCreate()
  }

  import spark.implicits._

    val fileName: String = "smalltest.txt" //Has 22 rows of data

    /** Load training and test data and cache it */
    val data = DataLoadOption.createDataFrame(fileName) match {
      case Some(df) => df
        .filter($"activityLabel" > 0)
        .withColumn("binaryLabel", when($"activityLabel".between(1, 3), 0).otherwise(1))
        .withColumn("uniqueID", monotonically_increasing_id())
      case None => throw new UnsupportedOperationException("Couldn't create DataFrame")
    }

    val N: Int = data.count().toInt

    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z") // 3 features selected
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

    val L = 10 // hiddenNodes
    val bias: BDV[Double] = BDV.rand[Double](L) // L x 1
    val weights: BDM[Double] = BDM.rand[Double](L, numFeatures) // L x numFeatures

    val X: BDM[Double] = extractFeaturesMatrix(dataWithFeatures) // numFeatures x N

    val T: BDV[Double] = extractLabelsMatrix(dataWithFeatures) // N

    val M = weights * X //L x N

    val H: BDM[Double] = sigmoid((M(::, *) + bias).t) // We want H to be N x L so that pinv(H) is L x N

    val beta: BDV[Double] = pinv(H) * T // (L x N).N => gives vector of length L

  /** Checks that the data structures passed into ELMModel are of the correct size when used in predictRaw() */
  test("[01] Data structures used in computations are of the correct size") {

    assert(bias.isInstanceOf[BDV[Double]])
    assert(bias.length == L)

    assert(weights.isInstanceOf[BDM[Double]])
    assert(weights.rows == L)
    assert(weights.cols == numFeatures)

    assert(X.isInstanceOf[BDM[Double]])
    assert(X.rows == numFeatures)
    assert(X.cols == N)

    assert(T.isInstanceOf[BDV[Double]])
    assert(T.length == N)

    assert(M.isInstanceOf[BDM[Double]])
    assert(M.rows == L)
    assert(M.cols == N)

    assert(H.isInstanceOf[BDM[Double]])
    assert(H.rows == N)
    assert(H.cols == L)

    assert(beta.isInstanceOf[BDV[Double]])
    assert(beta.length == L)
  }

  /**
    * This method only works when passing in a DataFrame and for a features vector of length N
    * Simple matrix multiplication: T = (Beta.t x H).t where H is calculated from the new dataset
    * In fact the features vector used in predictRaw() is for a single sample
    * So this method is not used in the production code
    */
  test("[02] Can return the labels in one go using T = Beta.H(transpose) with features calculated from DataFrame") {

    def predictRaw(features: DataFrame) :SDV = {

      val featuresArray: Array[Double] = features.rdd.flatMap(r => r.getAs[Vector](0).toArray).collect
      val featuresMatrix = new BDM[Double](numFeatures, N, featuresArray) //NB Features must be of size (numFeatures, N)

      val M = weights * featuresMatrix // L x numFeatures. numFeatures x N where N is no. of test samples.
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

  /**
    * In this test a feature vector is passed in from a single sample.
    * Simple matrix multiplication: T = (Beta.t x H).t where H is calculated from the dataset
    * for which predictions are being made
    * This is the method implemented in ELMModel.predictRaw()
    */
  test("[03] Can calculate T using pass in a features vector to predictRaw() using single AF") {

    def predictRaw(features: Vector): SDV = {

      val featuresArray: Array[Double] = features.toArray
      val featuresMatrix = new BDM[Double](numFeatures, 1, featuresArray)

      val M = weights * featuresMatrix // L x numFeatures. numFeatures x N where N is no. of test samples.

      val Z = M(::, *) + bias

      def calculateH(af: String, Z: BDM[Double]): BDM[Double] = af match {
        case "sigmoid" => sigmoid(Z)
        case "tanh" => tanh(Z)
        case "sin" => sin(Z)
        case _ => throw new IllegalArgumentException("Activation function must be sigmoid, tanh, or sin")
      }

      val H = sigmoid(M(::, *) + bias)// L x numFeatures
      val T = beta.t * H // L.(L x N) of type Transpose[DenseVector]

      new SDV(T.t.toArray)
    }

    val featuresRDD: RDD[Vector] = dataWithFeatures.select("features").rdd.map(_.getAs[Vector]("features"))

    val singleFeature: Vector = featuresRDD.collect()(0) //first feature in the array

    val singlePrediction: SDV = predictRaw(singleFeature) //uses sigmoid
    assert(singlePrediction.isInstanceOf[SDV])
    assert(singlePrediction.size == 1)
    println(singlePrediction.values.mkString(","))

    val predictions: SDV = new SDV(featuresRDD.collect().flatMap(f => predictRaw(f).toArray))
    assert(predictions.isInstanceOf[SDV])
    assert(predictions.size == data.count())

  }

    /**
    * In this test a feature vector is passed in from a single sample.
    * Simple matrix multiplication: T = (Beta.t x H).t where H is calculated from the dataset
    * for which predictions are being made
    * This is the method implemented in ELMModel.predictRaw()
    */
  test("[04] Can calculate T using a features vector to predictRaw() with different activation functions") {

    def predictRaw(features: Vector) :Tuple3[SDV, SDV, SDV] = {

      val featuresArray: Array[Double] = features.toArray
      val featuresMatrix = new BDM[Double](numFeatures, 1, featuresArray)

      val M = weights * featuresMatrix // L x numFeatures. numFeatures x N where N is no. of test samples.

      val Z = M(::, *) + bias

      def calculateH(af: String, Z: BDM[Double]): BDM[Double] = af match {
        case "sigmoid" => sigmoid(Z)
        case "tanh" => tanh(Z)
        case "sin" => sin(Z)
        case _ => throw new IllegalArgumentException("Activation function must be sigmoid, tanh, or sin")
      }

      val Hsig = calculateH("sigmoid", Z)
      val Htan = calculateH("tanh", Z)
      val Hsin = calculateH("sin", Z)

      val H = sigmoid(M(::, *) + bias) // L x numFeatures
      val T = beta.t * H // L.(L x N) of type Transpose[DenseVector]

      val Tsig = beta.t * Hsig
      val Ttan = beta.t * Htan
      val Tsin = beta.t * Hsin


      (new SDV((Tsig.t).toArray),new SDV((Ttan.t).toArray),new SDV((Tsin.t).toArray))
    }

    val featuresRDD: RDD[Vector] = dataWithFeatures.select("features").rdd.map(_.getAs[Vector]("features"))

    val singleFeature: Vector = featuresRDD.collect()(0) //first feature in the array

    val singlePrediction1: SDV = predictRaw(singleFeature)._1 //uses sigmoid
    assert(singlePrediction1.isInstanceOf[SDV])
    assert(singlePrediction1.size == 1)
    println(singlePrediction1.values.mkString(","))

    val predictions1: SDV = new SDV(featuresRDD.collect().flatMap(f => predictRaw(f)._1.toArray))
    assert(predictions1.isInstanceOf[SDV])
    assert(predictions1.size == data.count())

    val singlePrediction2: SDV = predictRaw(singleFeature)._2 //uses tanh
    assert(singlePrediction2.isInstanceOf[SDV])
    assert(singlePrediction2.size == 1)
    println(singlePrediction2.values.mkString(","))

    val predictions2: SDV = new SDV(featuresRDD.collect().flatMap(f => predictRaw(f)._2.toArray))
    assert(predictions2.isInstanceOf[SDV])
    assert(predictions2.size == data.count())


    val singlePrediction3: SDV = predictRaw(singleFeature)._3 //uses sin
    assert(singlePrediction3.isInstanceOf[SDV])
    assert(singlePrediction3.size == 1)
    println(singlePrediction3.values.mkString(","))

    val predictions3: SDV = new SDV(featuresRDD.collect().flatMap(f => predictRaw(f)._3.toArray))
    assert(predictions3.isInstanceOf[SDV])
    assert(predictions3.size == data.count())

  }

  /**
    * Another test using a udf, which passes - but no access to a DataFrame within ELMModel.
    * Therefore this code is not used
    */
  test("[05] using a Spark udf") {

    def extractUdf = udf((v: SDV) => v.toArray)

    val emptyDF = spark.emptyDataFrame

    val temp: DataFrame = dataWithFeatures.withColumn("extracted_features", extractUdf($"features"))

    temp.printSchema()

    val featuresArray1: Array[Double] = temp.rdd.map(r => r.getAs[Double](0)).collect
    val featuresArray2: Array[Double] = temp.rdd.map(r => r.getAs[Double](1)).collect
    val featuresArray3: Array[Double] = temp.rdd.map(r => r.getAs[Double](2)).collect

    val flatfeatures: Array[Double] = Array(featuresArray1, featuresArray2, featuresArray3).flatten

    temp.show()

    assert(featuresArray1.length == N)
    assert(featuresArray2.length == N)
    assert(featuresArray3.length == N)


    val featuresMatrix = new BDM[Double](N, numFeatures, flatfeatures) //gives matrix in column major order
    println(featuresMatrix.data.mkString(","))

    assert(featuresMatrix.rows == N)
    assert(featuresMatrix.cols == numFeatures)
  }

  /**
    * This test fails because the operation over the RDD is not serialisable.
    * Causes exception: calculateNode(f) not serializable. So test cancelled
    * This logic is not used in the model.
    */
  test("[05]. Can return a label for a single sample") {

    cancel("Test cancelled because RDD not serialisable")

    val features: RDD[Vector] = dataWithFeatures.select("features").rdd.map(r => r.getAs[Vector](0))

    def predictRaw(features: RDD[Vector]): SDV = {
      val outputArray: Array[Double] = features.map(f => calculateNode(f)).collect()
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
