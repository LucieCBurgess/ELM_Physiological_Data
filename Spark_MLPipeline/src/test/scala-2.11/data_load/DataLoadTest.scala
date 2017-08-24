package data_load

import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by lucieburgess on 24/08/2017.
  */
object DataLoadTest extends SparkSessionTestWrapper {

  import spark.implicits._

  /** Helper function to create a DataFrame from a textfile, re-used in subsequent tests */
  def createDataFrame(fileName: String) :DataFrame =  {

    val df = spark.sparkContext
      .textFile("/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/" + fileName)
      .map(_.split("\\t"))
      .map(attributes => mHealthUser(attributes(0).toDouble, attributes(1).toDouble, attributes(2).toDouble,
        attributes(3).toDouble, attributes(4).toDouble,
        attributes(5).toDouble, attributes(6).toDouble, attributes(7).toDouble,
        attributes(8).toDouble, attributes(9).toDouble, attributes(10).toDouble,
        attributes(11).toDouble, attributes(12).toDouble, attributes(13).toDouble,
        attributes(14).toDouble, attributes(15).toDouble, attributes(16).toDouble,
        attributes(17).toDouble, attributes(18).toDouble, attributes(19).toDouble,
        attributes(20).toDouble, attributes(21).toDouble, attributes(22).toDouble,
        attributes(23).toInt))
      .toDF()
      .cache()
    df
  }
}
