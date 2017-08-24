package data_load

/**
  * Created by lucieburgess on 13/08/2017.
  */

// FIXME - need to include unique ID and global temp view within production code

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.scalatest.{BeforeAndAfter, FunSuite}

class GlobalTempViewTest extends FunSuite with BeforeAndAfter with SparkSessionTestWrapper {

  import spark.implicits._

  val mHealthUser1DF: DataFrame = DataLoadTest.createDataFrame("firstline_subject1.txt")
  val mHealthUser2DF: DataFrame = DataLoadTest.createDataFrame("firstline_subject2.txt")
  val mHealthUser3DF: DataFrame = DataLoadTest.createDataFrame("firstline_subject3.txt")

  val mHealthJoined: DataFrame = mHealthUser1DF.union(mHealthUser2DF).union(mHealthUser3DF)

  /** Add unique rowID, which will be unique but not consecutive */
  val mHealthIndexed: DataFrame = mHealthJoined.withColumn("uniqueID", monotonically_increasing_id())

  /** Create global temporary view which can be re-used across Spark Sessions */
  mHealthIndexed.createGlobalTempView("mHealthIndexed_GT")

  /** Check the data can be returned from a given row ID from the global temp view */
  test("[03] correct attribute can be selected from global temp view]") {

    val result: Dataset[Double] = spark.sql("SELECT acc_Chest_Y FROM global_temp.mHealthIndexed_GT WHERE uniqueID = 17179869184").map(x => x.getAs[Double]("acc_Chest_Y"))

    assertResult(0.2986) {
      result.collect()(0)
    }
  }

  /** Print the schema and check it conforms with the case class by inspection */
  test("[04] schema prints correctly") {
    mHealthIndexed.printSchema()
  }

  /** Global temp view persists to new Spark Session */
  test("[05] global temp view persists") {

    var result2 :Dataset[Double] = spark.newSession().sql("SELECT acc_Chest_Y FROM global_temp.mHealthIndexed_GT WHERE uniqueID = 17179869184").map(x => x.getAs[Double]("acc_Chest_Y"))

    assertResult(0.2986) {
      result2.collect()(0)
    }
  }

}
