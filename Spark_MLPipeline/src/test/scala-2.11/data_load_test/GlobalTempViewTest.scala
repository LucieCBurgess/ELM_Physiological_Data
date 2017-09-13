package data_load_test

/**
  * Created by lucieburgess on 13/08/2017.
  * Updated tests in include wrapped version of file using Option and took out multiple file load,
  * so class is testing the temp view only
  * Also added a monotonically increasing id()
  * ALL TESTS PASS
  */

import dev.data_load.{DataLoadOption, MHealthUser}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.scalatest.FunSuite

class GlobalTempViewTest extends FunSuite {

  val spark: SparkSession = {
    SparkSession.builder().master("local[4]").appName("GlobalTempViewTest").getOrCreate()
  }

  import spark.implicits._

  val fileName = "mHealth_subject1.txt"

  val df: Option[DataFrame] = scala.util.Try(spark.sparkContext.textFile("/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/" + fileName)
      .map(_.split("\\t"))
      .map(attributes => MHealthUser(attributes(0).toDouble, attributes(1).toDouble, attributes(2).toDouble,
        attributes(3).toDouble, attributes(4).toDouble,
        attributes(5).toDouble, attributes(6).toDouble, attributes(7).toDouble,
        attributes(8).toDouble, attributes(9).toDouble, attributes(10).toDouble,
        attributes(11).toDouble, attributes(12).toDouble, attributes(13).toDouble,
        attributes(14).toDouble, attributes(15).toDouble, attributes(16).toDouble,
        attributes(17).toDouble, attributes(18).toDouble, attributes(19).toDouble,
        attributes(20).toDouble, attributes(21).toDouble, attributes(22).toDouble,
        attributes(23).toInt))
      .toDF()
      .cache()).toOption

  val df2: DataFrame = df match {
    case Some(df) => df.withColumn("uniqueID", monotonically_increasing_id())
    case None => throw new UnsupportedOperationException("Couldn't create DataFrame")
  }

  df2.show()

  val df3: Unit = df2.createGlobalTempView("mHealth1GTV")

  /** Check the data can be returned from a given row ID from the global temp view */
  test("[03] correct attribute can be selected from global temp view]") {

    val result: Dataset[Double] = spark.sql("SELECT acc_Chest_Y FROM global_temp.mHealth1GTV WHERE uniqueID = 3").map(x => x.getAs[Double]("acc_Chest_Y"))

    assertResult(0.21422) {
      result.collect()(0)
    }
  }

  /** Print the schema and check it conforms with the case class by inspection */
  test("[04] schema prints correctly") {
    df2.printSchema()
  }

  /** Global temp view persists to new Spark Session */
  test("[05] global temp view persists") {

    var result2 :Dataset[Double] = spark.newSession().sql("SELECT acc_Chest_Y FROM global_temp.mHealth1GTV WHERE uniqueID = 3").map(x => x.getAs[Double]("acc_Chest_Y"))

    assertResult(0.21422) {
      result2.collect()(0)
    }
  }
}
