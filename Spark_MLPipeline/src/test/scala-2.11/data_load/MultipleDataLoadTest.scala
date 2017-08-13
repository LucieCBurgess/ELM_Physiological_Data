package data_load

/**
  * Created by lucieburgess on 13/08/2017.
  */

import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions._
import org.scalatest.{BeforeAndAfterEach, FunSuite, BeforeAndAfter}

case class mHealthUser(acc_Chest_X: Double, acc_Chest_Y: Double, acc_Chest_Z: Double,
                        ecg_1: Double, ecg_2: Double,
                        acc_Ankle_X: Double, acc_Ankle_Y: Double, acc_Ankle_Z: Double,
                        gyro_Ankle_X: Double, gyro_Ankle_Y: Double, gyro_Ankle_Z: Double,
                        magno_Ankle_X: Double, magno_Ankle_Y: Double, magno_Ankle_Z: Double,
                        acc_Arm_X: Double, acc_Arm_Y: Double, acc_Arm_Z: Double,
                        gyro_Arm_X: Double, gyro_Arm_Y: Double, gyro_Arm_Z: Double,
                        magno_Arm_X: Double, magno_Arm_Y: Double, magno_Arm_Z: Double,
                        activityLabel: Int)

class MultipleDataLoadTest extends FunSuite with BeforeAndAfterEach {

  private val master = "local[*]"
  private val appName = "multiple_data_load_test"

  var spark: SparkSession = _

  override def beforeEach() {
    spark = new SparkSession.Builder().appName(appName).master(master).getOrCreate()
  }

  test("[01] Loading data from more than one file") {

    val sqlContext = spark.sqlContext
    import sqlContext.implicits._

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

    mHealthJoined.show()

    assert(mHealthJoined.count() == 3)

    /** Add unique rowID, which will be unique but not consecutive */
    val mHealthIndexed = mHealthJoined.withColumn("uniqueID", monotonically_increasing_id())

    /** Create global temporary view which can be re-used across Spark Sessions */
    mHealthIndexed.createGlobalTempView("mHealthIndexed")

    /** Print the scheme and check it conforms with the case class by inspection */
    mHealthIndexed.printSchema()

    val result: Dataset[Double] = spark.sql("SELECT acc_Chest_Y FROM global_temp.mHealthIndexed WHERE uniqueID = 17179869184").map(x => x.getAs[Double]("acc_Chest_Y"))

    assertResult(0.2986) {
      result.collect()(0)
    }

    /** write to output tab delimited csv file */
    //mHealthIndexed.write.format("csv").option("delimiter","\\t").option("path", "/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/TestFolder").saveAsTable("mHealthIndexed")

    /** Check data loads again */
    //FIXME - doesn't work
    val mHealthAllDF = spark.sparkContext
      .textFile("/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/TestFolder/mHealthIndexed.csv")
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

    mHealthAllDF.show()
  }

  override def afterEach() = {
    spark.stop()
  }
}