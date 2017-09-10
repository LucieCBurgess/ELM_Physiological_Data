package dev.data_load

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{monotonically_increasing_id, when}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.util.Try

/**
  * Created by lucieburgess on 10/09/2017. Try using WholeTextFile method instead of TextFile method.
  */
object DataLoadWTF {

  lazy val spark: SparkSession = SparkSession
    .builder()
    .master("local[4]")
    .appName("Data_load")
    .getOrCreate()

  def main(args: Array[String]): Unit = {

    val folderName: String = "Multiple"

    val data = createDataFrame(folderName)
        //.filter($"activityLabel" > 0.0)
        //.withColumn("binaryLabel", when($"activityLabel".between(1.0, 3.0), 0.0).otherwise(1.0))
        //.withColumn("uniqueID", monotonically_increasing_id())

    val Nsamples: Int = data.count().toInt
    println(s"The number of training samples is $Nsamples")
  }

  def createDataFrame(folderName: String): DataFrame = {

    import spark.implicits._

    val data: RDD[(String, String)] = spark.sparkContext.wholeTextFiles("/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/" + folderName, 5)

    val df = data.map {case (fileName, content) => content }.flatMap(_.split("\n")).map(_.split("\\t"))
          .map(attributes => MHealthUser(attributes(0).toDouble, attributes(1).toDouble, attributes(2).toDouble,
          attributes(3).toDouble, attributes(4).toDouble,
          attributes(5).toDouble, attributes(6).toDouble, attributes(7).toDouble,
          attributes(8).toDouble, attributes(9).toDouble, attributes(10).toDouble,
          attributes(11).toDouble, attributes(12).toDouble, attributes(13).toDouble,
          attributes(14).toDouble, attributes(15).toDouble, attributes(16).toDouble,
          attributes(17).toDouble, attributes(18).toDouble, attributes(19).toDouble,
          attributes(20).toDouble, attributes(21).toDouble, attributes(22).toDouble,
          attributes(23).toInt.toDouble))
        .toDF()
        .cache()
      df
  }
}