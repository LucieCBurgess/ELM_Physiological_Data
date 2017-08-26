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

// FIXME - configurations need to be set at some point - within val spark .config("spark.some.config.option", "some-value")

object MultipleDataLoad extends SparkSessionWrapper {

  def main(args: Array[String]) {

    import spark.implicits._

    /** Helper function to create a DataFrame from a textfile */
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
      df
    }

    /** Helper function to omit unlabelled data */
    def omitUnLabelledData(df: DataFrame) :DataFrame = {

      val labelledDF = df.filter($"activityLabel" =!= 0)
      labelledDF
    }

    val fileNames: Seq[String] = Seq("mHealth_subject1.txt", "mHealth_subject2.txt", "mHealth_subject3.txt", "mHealth_subject4.txt",
                              "mHealth_subject5.txt", "mHealth_subject6.txt", "mHealth_subject7.txt", "mHealth_subject8.txt",
                              "mHealth_subject9.txt", "mHealth_subject10.txt")

    val dfSequence: Seq[DataFrame] = fileNames.map(f => createDataFrame(f)).map(f => omitUnLabelledData(f))

    val completeDF: DataFrame = dfSequence.reduce((df1, df2) => df1.union(df2))

    completeDF.show(2, 0)
    println("The number of rows in the dataset is " + completeDF.count()) //343195
    completeDF.groupBy("activityLabel").count().show()
  }
  spark.stop()
}