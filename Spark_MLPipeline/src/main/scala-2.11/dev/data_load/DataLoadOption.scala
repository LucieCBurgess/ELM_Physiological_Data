package dev.data_load

import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.util.Try

/**
  * Created by lucieburgess on 24/08/2017.
  * Helper function to create a DataFrame from a textfile, re-used in subsequent classes, with some exception handling
  * See https://stackoverflow.com/questions/45895642/how-to-correctly-handle-option-in-spark-scala
  */

//FIXME - there should be an elegant way to validate that the incoming data is in the correct format, e.g. there aren't Strings or Longs in the data instead of Doubles and Ints
// For a good example see https://stackoverflow.com/questions/33270907/how-to-validate-contents-of-spark-dataframe

object DataLoadOption {
  lazy val spark: SparkSession = SparkSession
    .builder()
    .master("local[4]")
    .appName("Data_load")
    .getOrCreate()

  def createDataFrame(fileName: String): Option[DataFrame] = {

    import spark.implicits._

    val df: Option[DataFrame] = Try(spark.sparkContext.textFile("/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/" + fileName)
      .map(_.split("\\t"))
      .map(attributes => MHealthUser(attributes(0).toDouble, attributes(1).toDouble, attributes(2).toDouble,
        attributes(3).toDouble, attributes(4).toDouble,
        attributes(5).toDouble, attributes(6).toDouble, attributes(7).toDouble,
        attributes(8).toDouble, attributes(9).toDouble, attributes(10).toDouble,
        attributes(11).toDouble, attributes(12).toDouble, attributes(13).toDouble,
        attributes(14).toDouble, attributes(15).toDouble, attributes(16).toDouble,
        attributes(17).toDouble, attributes(18).toDouble, attributes(19).toDouble,
        attributes(20).toDouble, attributes(21).toDouble, attributes(22).toDouble,
        attributes(23).toDouble))
      .toDF()
      .cache()).toOption
    df
  }
}
