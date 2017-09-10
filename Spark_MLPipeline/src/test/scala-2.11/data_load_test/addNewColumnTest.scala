package data_load_test

/**
  * Created by lucieburgess on 23/08/2017.
  * Unit test for adding a column with a conditional statement, changing the ActivityLevels from 0-12 to 0 \ 1
  * Checks that the total number of indexed labels (0 \ 1) sums to the total size of the dataset
  * Unit test constructed for mHealth_subject1 only
  * ALL TESTS PASS
  */

import dev.data_load.DataLoadOption
import org.apache.spark.sql.{Column, DataFrame}
import org.scalatest.FunSuite
import org.apache.spark.sql.functions._


class AddNewColumnTest extends FunSuite with SparkSessionTestWrapper {

  import spark.implicits._

  val dfs : Seq[DataFrame] = Seq("mHealth_subject1.txt").map(DataLoadOption.createDataFrame).flatten

  val df: DataFrame = dfs.head.filter($"activityLabel" > 0)

  /** Test to add a new column to the dataframe with indexed label (0 or 1) from activityLabel (1-3, 4-12) */
  test("[01] Adding a new column to the DataFrame with indexedLabel") {

    val newCol: Column = when($"activityLabel".between(1, 3), 0).otherwise(1)
    val df2: DataFrame = df.withColumn("indexedLabel", newCol)

    df2.show()

    df2.groupBy("indexedLabel").count.orderBy("indexedLabel").show
  }

  /** Checks the activityLabel groupBy statement above tallies to sum of activityLabels in correct range */
  test("[02] Check this corresponds with the correct sum of activityLabels in range") {

    /**
      * A helper function to sum the number of activityLabels within a given range to use in testing
      * @param df the DataFrame to be operated on
      * @return a new DataFrame with sum of number of records with label 1-3(inactive) and records with label 4-12(active)
      */
    def createRangeActivityLabels(df: DataFrame): List[Long] = {

      val activityRange: List[(Int, Int)] = List((1, 3), (4, 12))

      val sumInRange: List[Column] = activityRange.map {
        case (x, y) =>
          val newLabel = s"${x}_${y}"
          sum(when($"activityLabel".between(x, y), 1).otherwise(0)).alias(newLabel)
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

    val result = createRangeActivityLabels(df)

    val dfSize = df.count()

    assertResult(9216) { result.head }
    assertResult(25958) { result(1) }
    assert((result.head + result(1)).equals(dfSize)) //35174

  }
}


