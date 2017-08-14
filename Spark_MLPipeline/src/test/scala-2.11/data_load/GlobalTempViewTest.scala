package data_load

import org.apache.spark.sql.{Dataset}
import org.apache.spark.sql.functions._
import org.scalatest.{BeforeAndAfter, FunSuite}

/**
  * Created by lucieburgess on 13/08/2017.
  */

class GlobalTempViewTest extends FunSuite with BeforeAndAfter with SparkSessionTestWrapper {

  import spark.implicits._

  val mHealthUser1DF = spark.sparkContext
    .textFile("/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/firstline_subject1.txt")
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

  val mHealthUser2DF = spark.sparkContext
    .textFile("/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/firstline_subject2.txt")
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

  val mHealthUser3DF = spark.sparkContext
    .textFile("/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/firstline_subject3.txt")
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

  val mHealthJoined = mHealthUser1DF.union(mHealthUser2DF).union(mHealthUser3DF)

  /** Add unique rowID, which will be unique but not consecutive */
  val mHealthIndexed = mHealthJoined.withColumn("uniqueID", monotonically_increasing_id())

  /** Create global temporary view which can be re-used across Spark Sessions */
  mHealthIndexed.createGlobalTempView("mHealthIndexed_GT")

  /** Check the data can be returned from a given row ID from the global temp view */
  test("[02 correct attribute can be selected from global temp view]") {

    val result: Dataset[Double] = spark.sql("SELECT acc_Chest_Y FROM global_temp.mHealthIndexed_GT WHERE uniqueID = 17179869184").map(x => x.getAs[Double]("acc_Chest_Y"))

    assertResult(0.2986) {
      result.collect()(0)
    }
  }

  /** Print the scheme and check it conforms with the case class by inspection */
  test("[03 schema prints correctly") {
    mHealthIndexed.printSchema()
  }

  test("[04 global temp view persists") {

    var result2 :Dataset[Double] = spark.newSession().sql("SELECT acc_Chest_Y FROM global_temp.mHealthIndexed_GT WHERE uniqueID = 17179869184").map(x => x.getAs[Double]("acc_Chest_Y"))

    assertResult(0.2986) {
      result2.collect()(0)
    }
  }

}
