package dev.data_load

import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.util.Try

/**
  * Created by lucieburgess on 04/08/2017.
  * Creates a Spark dataframe for each mHealth user, filters out unlabelled data and joins the results to create a single DF.
  * All data in the mHealth dataset are doubles apart from activity_Label which is an integer. Schema inferred from case class mHealthUser.
  *
  * Code is tested in a separate package of the same name in test
  *
  * From: https://spark.apache.org/docs/latest/sql-programming-guide.html#getting-started
  * A dataset is a distributed collection of data. Dataset is a new interface added in Spark 1.6. The API is available in Scala and Java
  * A Dataframe is a dataset organised into named columns. A DataFrame is represented by a Dataset of Rows.
  * In the Scala API, DataFrame is simply a type alias of Dataset[Row].
  *
  * From https://spark.apache.org/docs/latest/sql-programming-guide.html#data-sources
  * The Scala interface for Spark SQL supports automatically converting an RDD containing case classes to a DataFrame
  * The case class defines the scheme of the table
  * The names of the arguments to the case classes are read using reflection and become the names of the columns
  * So we define the mHealth_subject1.txt file schema using a case class and infer it using reflection, as shown below
  */

object DataLoad {

  lazy val spark: SparkSession = SparkSession
    .builder()
    .master("local[4]")
    .appName("Data_load_using_single_files_plus_union")
    .getOrCreate()

  import spark.implicits._

    /** Helper function to create a DataFrame from a textfile and infer the schema by reflection */
    def createDataFrame(fileName: String) :DataFrame =  {

      val df = spark.sparkContext
        .textFile("/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/" + fileName)
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
      df
    }
}