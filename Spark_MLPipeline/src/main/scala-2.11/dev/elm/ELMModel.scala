package dev.elm

import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.ml.linalg.{Vector, DenseMatrix => SDM, DenseVector => SDV}
import breeze.linalg.{*, pinv, DenseMatrix => BDM, DenseVector => BDV, Matrix => BM, Vector => BV}
import breeze.numerics._
import dev.data_load.SparkSessionWrapper
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._

/**
  * Created by lucieburgess on 27/08/2017.
  * This class is a Transformer according to the Spark Pipeline model, and extends ClassificationModel
  * The Classifier is trained with ELMClassifier from the algorithm ELMClassifierAlgo
  *
  * This uses the default implementation of transform(), which reads column "features" and
  * outputs columns "prediction". 'rawPrediction" is calculated for each labelled point in the dataset
  * from the parameters passed into ELMClassifier, based on the ELMClassifier trained using column "features".
  * It also uses the default implementation of transformSchema(schema: StructType).
  *
  * This uses the default implementation of predict(), which chooses the label corresponding to
  * the maximum value returned by [[predictRaw()]], as stated in the Classifier Model API
  *
  */
class ELMModel(override val uid: String, val modelWeights: BDM[Double], val modelBias: BDV[Double], val modelBeta: BDV[Double],
               val modelHiddenNodes: Int, val modelAF: String, val modelNumFeatures: Int)
  extends ClassificationModel[Vector, ELMModel]
    with ELMParams with DefaultParamsWritable {

  //import spark.implicits._

  override def copy(extra: ParamMap): ELMModel = {
    val copied = new ELMModel(uid, modelWeights, modelBias, modelBeta, modelHiddenNodes, modelAF, modelNumFeatures)
    copyValues(copied, extra).setParent(parent)
  }

  /** Number of classes the label can take. 2 indicates binary classification */
  override val numClasses: Int = 2

  val inputColName = "features"
  val outputColName = "prediction"

  /**
    * @param features, the vector of features being input into the model
    * @return vector where element i is the raw prediction for label i. This raw prediction may be any real number,
    *         where a larger value indicates greater confidence for that label
    *         The underlying method in ClassificationModel then predicts raw2prediction, which given a vector of raw predictions
    *         selects the predicted labels. raw2prediction can be overridden to support thresholds which favour particular labels.
    *         NB. data.select("features") gives an instance of a dataframe, so this is the type of the features column
    */
    //FIXME - re-write this function to take a Vector parameter
    // Problem is that featuresArray is not properly stripping out the features. Need to change FeaturesType to something we
    // can actually pass in. This might involve re-writing transform instead, as it takes a dataset
  override def predictRaw(features: Vector) :Vector = {

      //val numSamples = 9
      val featuresArray = features.toArray
      val featuresMatrix = new BDM[Double](modelNumFeatures, 1, featuresArray) //numFeatures x 1

      val bias: BDV[Double] = modelBias //L x 1
      val weights: BDM[Double] = modelWeights // Lx numFeatures
      val beta: BDV[Double] = modelBeta // (LxN).N => gives vector of length L

      val M = weights * featuresMatrix // L x numFeatures. numFeatures x 1 = L x 1
      val H = sigmoid((M(::, *)) + bias) // L x 1
      val T = beta.t * H // L.(L x 1) of type Transpose[DenseVector]
      new SDV((T.t).toArray) // length 1
  }


//  override def transform(data: Dataset[_]): DataFrame = {
//
//    import data.sparkSession.implicits._
//
//    val outputSchema = transformSchema(data.schema, logging = true)
//    val inputType = data.schema("activityLabel").dataType // this should really be a double not an int
//
//    val numSamples: Int = data.count().toInt
//
//    def extractUdf = udf((v: SDV) => v.toArray)
//    val temp: DataFrame = data.withColumn("extracted_features", extractUdf($"features"))
//
//    temp.printSchema()
//
//    val featuresArray1: Array[Double] = temp.rdd.map(r => r.getAs[Double](0)).collect
//    val featuresArray2: Array[Double] = temp.rdd.map(r => r.getAs[Double](1)).collect
//    val featuresArray3: Array[Double] = temp.rdd.map(r => r.getAs[Double](2)).collect
//
//    val allfeatures: Array[Array[Double]] = Array(featuresArray1, featuresArray2, featuresArray3)
//    val numFeatures: Int = allfeatures.length
//
//    val flatFeatures: Array[Double] = allfeatures.flatten
//
//    temp.select("features","extracted_features").show(10)
//
//    val featuresMatrix = new BDM[Double](numSamples, numFeatures, flatFeatures) //gives matrix in column major order
//
//    val bias: BDV[Double] = modelBias // L x 1
//    val weights: BDM[Double] = modelWeights //  L x numFeatures
//    val beta: BDV[Double] = modelBeta // L x N.N => gives vector of length L
//
//    val M = weights * featuresMatrix // L x numFeatures. numFeatures x N where N is no. of test samples. NB Features must be of size (numFeatures, N)
//    val H = sigmoid((M(::, *)) + bias) // L x numFeatures
//    val T = beta.t * H // L.(L x N) of type Transpose[DenseVector]
//    val rdd: RDD[Double] = spark.sparkContext.parallelize((T.t).toArray)//length N
//
//    val output: RDD[Row] = data.rdd.zip(rdd).map(r => Row.fromSeq(Seq(r._1) ++ Seq(r._2)))
//
//    spark.createDataFrame(output, data.schema)
//  }

}
