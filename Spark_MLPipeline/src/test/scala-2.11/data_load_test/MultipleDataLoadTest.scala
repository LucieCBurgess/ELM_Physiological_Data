package data_load_test

/**
  * Created by lucieburgess on 13/08/2017.
  * Tests data load of a single line from three files and joins the results.
  * NB. .map(DataLoadTest.createDataFrame) deals with the Option[DataFrame] problem
  * See https://stackoverflow.com/questions/45895642/how-to-correctly-handle-option-in-spark-scala
  */

import dev.data_load.DataLoad
import org.apache.spark.sql.DataFrame
import org.scalatest.FunSuite

class MultipleDataLoadTest extends FunSuite with SparkSessionTestWrapper {

  val dfs : Seq[DataFrame] = Seq("firstline_subject1.txt","firstline_subject2.txt","firstline_subject3.txt").map(DataLoad.createDataFrame).flatten

  test("[02A] Loading data to create separate data frames") {

    val mHealthUser1DF = dfs.head
    val mHealthUser2DF = dfs(1)
    val mHealthUser3DF = dfs(2)

    assert(mHealthUser1DF.count() == 1 && mHealthUser2DF.count() == 1 && mHealthUser3DF.count() == 1)
  }

  test("[02B] Joining data frames creates one data frame of correct size") {

    val mHealthUser1DF = dfs(0)
    val mHealthUser2DF = dfs(1)
    val mHealthUser3DF = dfs(2)

    val mHealthJoined = mHealthUser1DF.union(mHealthUser2DF).union(mHealthUser3DF)
    mHealthJoined.show()
    assert(mHealthJoined.count() == 3)
  }
}