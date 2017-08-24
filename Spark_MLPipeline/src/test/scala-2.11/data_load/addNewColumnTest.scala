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

  //FIXME createDataFrame method used ubiquitously so should be moved to a new class
  def createDataFrame(fileName: String): DataFrame = {

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

  val mHealthUser1DF: DataFrame = createDataFrame("mHealth_subject1.txt")

  /** Test to add a new column to the dataframe with indexed label (0 or 1) from activityLabel (1-3, 4-12) */
  test("[06] Adding a new column to the DataFrame with indexedLabel") {

    val newCol: Column = when($"activityLabel".between(1, 3), 0).otherwise(1)
    val df2: DataFrame = mHealthUser1DF.withColumn("indexedLabel", newCol)

    df2.show()

    df2.groupBy("indexedLabel").count.orderBy("indexedLabel").show
  }

  /** Checks the activityLabel groupBy statement above tallies to sum of activityLabels in correct range */
  test("[07] Check this corresponds with the correct sum of activityLabels in range") {

    /**
      * A helper function to sum the number of activityLabels within a given range to use in testing
      * @param df the DataFrame to be operated on
      * @return a new DataFrame with sum of number of records with label 1-3(inactive) and records with label 4-12(active)
      */
    def createRangeActivityLabels(df: DataFrame): List[Long] = {

      val activityRange: List[(Int, Int)] = List((1, 3), (4, 12))

      val sumInRange: List[Column] = activityRange.map {
        case (x, y) => {
          val newLabel = s"${x}_${y}"
          sum(when($"activityLabel".between(x, y), 1).otherwise(0)).alias(newLabel)
        }
      }

      // Needed to prevent aggregating twice within same function which causes an error
      val columnNames: List[Column] = activityRange.map {
        case (x, y) => $"${x}_${y}"
      }

      val df3: DataFrame = df.groupBy($"activityLabel").agg(sumInRange.head, sumInRange.tail: _*).orderBy($"activityLabel")
      df3.show

      val indexedLabel0 = df3.agg(sum(columnNames.head)).first.getAs[Long](0)
      val indexedLabel1 = df3.agg(sum(columnNames(1))).first.getAs[Long](0)

      val result: List[Long] = List(indexedLabel0, indexedLabel1)
      result
    }

    val result = createRangeActivityLabels(mHealthUser1DF)

    assertResult(9216) { result(0) }
    assertResult(25958) { result(1) }

  }
}


