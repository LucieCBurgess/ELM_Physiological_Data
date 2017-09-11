package data_load_test

/**
  * Created by lucieburgess on 13/08/2017.
  * Tests data load from a number of files and joins the results into a large DataFrame using Union.
  * Time to construct complete DataFrame is 1.752 seconds
  * ALL TESTS PASS
  */

import dev.data_load.{DataLoadOption, DataLoad}
import org.apache.spark.sql.DataFrame
import org.scalatest.FunSuite

class DataLoadUnionTest extends FunSuite with SparkSessionDataLoadTestWrapper {

  import spark.implicits._

  val smallFiles: Seq[String] = Seq("firstline_subject1.txt", "firstline_subject2.txt", "firstline_subject3.txt")

  val largeFiles: Seq[String] = Seq("mHealth_subject1.txt", "mHealth_subject2.txt", "mHealth_subject3.txt", "mHealth_subject4.txt",
    "mHealth_subject5.txt", "mHealth_subject6.txt", "mHealth_subject7.txt", "mHealth_subject8.txt",
    "mHealth_subject9.txt", "mHealth_subject10.txt")


  test("[01] Loading data to create separate data frames with small test files") {

    val dfs: Seq[DataFrame] = smallFiles.map(f => DataLoad.createDataFrame(f))

    val mHealthUser1DF = dfs.head
    val mHealthUser2DF = dfs(1)
    val mHealthUser3DF = dfs(2)

    assert(mHealthUser1DF.count() == 1 && mHealthUser2DF.count() == 1 && mHealthUser3DF.count() == 1)
  }

  test("[02] Using union creates one small dataframe of correct size") {

    val dfs: Seq[DataFrame] = smallFiles.map(f => DataLoad.createDataFrame(f))

    val mHealthUser1DF = dfs(0)
    val mHealthUser2DF = dfs(1)
    val mHealthUser3DF = dfs(2)

    val mHealthJoined = mHealthUser1DF.union(mHealthUser2DF).union(mHealthUser3DF)
    mHealthJoined.show()
    assert(mHealthJoined.count() == 3)
  }

  /** Loads the files, filters and then creates one large file using union. Takes 1.752 seconds
    * Checks that the size of the constructed DF is the same as the sum of the constituent DFs.
    */
  test("[03] Loading data to create separate dataframes and using Union for large files") {

    val start = System.nanoTime()

    val dfs: Seq[DataFrame] = largeFiles.map(f => DataLoad.createDataFrame(f)).map(df => df.filter($"activityLabel" > 0.0))

    val completeDF: DataFrame = dfs.reduce((df1, df2) => df1.union(df2))

    val finish = (System.nanoTime() - start) / 1e9

    println(s"Time to construct complete DataFrame: $finish seconds")

    completeDF.show()
    completeDF.orderBy("activityLabel").groupBy("activityLabel").count().show()

    assert(completeDF.count() == 343195) //The number of total labelled samples across all 10 users

    assert(completeDF.count.equals(dfs.map(d => d.count()).reduce(_+_)))

  }
}