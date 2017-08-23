package data_load

/**
  * Created by lucieburgess on 23/08/2017.
  * Unit test for adding a column with a conditional statement, changing the ActivityLevels from 0-12 to 0 \ 1
  */

import org.apache.spark.sql.{Column, DataFrame, Row}
import org.scalatest.FunSuite
import org.apache.spark.sql.functions._


class addNewColumnTest extends FunSuite with SparkSessionTestWrapper {

  import spark.implicits._

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
      .filter($"activityLabel" > 0)
    df
  }

  /**
    * Function which adds a new column and turns labels 1-4 to Inactive (==0) and labels 5-12 to Active (==1)
    *  @param df the DataFrame to which the new column is to be added
    * @param column, the column of activity labels which is to be converted from mutinomial to binomial
    * @return the new DataFrame with the added column
    *
    */
  // FIXME - this is not adding a new column, just replacing the existing column
  def createBinaryColumn(df: DataFrame, column: String): DataFrame = {

    val newCol: Column = when(col(column).equalTo(1 | 2 | 3 | 4), 0).otherwise(1).alias("indexedLabel")
    val df2: DataFrame = df.withColumn(column, newCol)
    df2
  }

  /**
    * //FIXME - this is not working as expected. See results from running the test
    * A helper function to sum the number of activityLabels within a given range to use in testing
    * @param df the DataFrame to be operated on
    * @return a new DataFrame with sum of number of records with label 1-4 and number of records with label 5-12
    */

  def createRangeActivityLabels(df: DataFrame): Unit = {

    val activityList: List[(Int, Int)] = List((1, 3), (4, 12))

    val exprs: List[Column] = activityList.map {
      case (x, y) => {
        val newLabel = s"${x}_${y}"
        sum(when($"activityLabel".between(x, y), 0).otherwise(1)).alias(newLabel)
      }
    }

    val df2: DataFrame = df.groupBy($"activityLabel").agg(exprs.head, exprs.tail: _*)
    val indexedLabelSums = df2.agg(sum(exprs.head), sum(exprs(1))).first

    println(s"************* Number of Inactive Labels: ${indexedLabelSums.getAs[Int](exprs.head.toString())} " +
      s"Number of Active Labels: ${indexedLabelSums.getAs[Int](exprs(1).toString())} ************* ")

  }

  test("[06] Loading data to create separate data frames") {

    val mHealthUser1DF: DataFrame = createDataFrame("mHealth_subject1.txt")

    //val newDF: DataFrame = createBinaryColumn(mHealthUser1DF, "activityLabel")

    val newCol: Column = when($"activityLabel".between(1,3), 0).otherwise(1)
    val df2: DataFrame = mHealthUser1DF.withColumn("indexedLabel", newCol)

    df2.show()
    df2.groupBy("indexedLabel").count.show

//    newDF.show()
//    newDF.groupBy("activityLabel").count().show()

    // mHealthUser1DF.groupBy("activityLabel").count().show()

    createRangeActivityLabels(mHealthUser1DF)

  }
}
