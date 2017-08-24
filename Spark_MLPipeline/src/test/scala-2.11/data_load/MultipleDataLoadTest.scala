package data_load

/**
  * Created by lucieburgess on 13/08/2017.
  * Tests data load of a single line from three files and joins the results.
  */

import org.apache.spark.sql.DataFrame
import org.scalatest.FunSuite

class MultipleDataLoadTest extends FunSuite with SparkSessionTestWrapper {

  val mHealthUser1DF: DataFrame = DataLoadTest.createDataFrame("firstline_subject1.txt")
  val mHealthUser2DF: DataFrame = DataLoadTest.createDataFrame("firstline_subject2.txt")
  val mHealthUser3DF: DataFrame = DataLoadTest.createDataFrame("firstline_subject3.txt")

  test("[02A] Loading data to create separate data frames") {

    mHealthUser1DF.show()
    mHealthUser2DF.show()
    mHealthUser3DF.show()

    assert(mHealthUser1DF.count() == 1 && mHealthUser2DF.count() == 1 && mHealthUser3DF.count() == 1)
  }

  test("[02B] Joining data frames creates one data frame of correct size") {

    val mHealthJoined = mHealthUser1DF.union(mHealthUser2DF).union(mHealthUser3DF)
    mHealthJoined.show()
    assert(mHealthJoined.count() == 3)
  }
}