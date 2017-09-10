package dev.data_load

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import scala.util.Try

/**
  * Created by lucieburgess on 10/09/2017. Try using WholeTextFile method instead of TextFile method.
  * Has an additional operation to flatmap the rows.
  * 3 partitions used in the method but could vary this to test the impact on performance.
  */
object DataLoadWTF {

  lazy val spark: SparkSession = SparkSession
    .builder()
    .master("local[4]")
    .appName("Data_load_using_whole_text_file")
    .getOrCreate()

  def createDataFrame(folderName: String): DataFrame = {

    import spark.implicits._

    val data: Option[RDD[(String, String)]] = Try(spark.sparkContext.wholeTextFiles("/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/" + folderName, 3)).toOption

    val df: DataFrame = data match {
      case Some(data) => data.map { case (fileName, content) => content }.flatMap(_.split("\n")).map(_.split("\\t"))
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
      case None => throw new UnsupportedOperationException("Creating the dataframe failed")
    }
    df
  }
}