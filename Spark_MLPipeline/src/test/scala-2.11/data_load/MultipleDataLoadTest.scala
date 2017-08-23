package data_load

/**
  * Created by lucieburgess on 13/08/2017.
  * Tests data load of a single line from three files and joins the results.
  */

import org.apache.spark.sql.DataFrame
import org.scalatest.{BeforeAndAfterEach, FunSuite}

class MultipleDataLoadTest extends FunSuite with BeforeAndAfterEach with SparkSessionTestWrapper {

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
    df
  }

  test("[02] Loading data to create separate data frames") {

    val mHealthUser1DF: DataFrame = createDataFrame("firstline_subject1.txt")
    val mHealthUser2DF: DataFrame = createDataFrame("firstline_subject2.txt")
    val mHealthUser3DF: DataFrame = createDataFrame("firstline_subject3.txt")

    mHealthUser1DF.show()
    mHealthUser2DF.show()
    mHealthUser3DF.show()

    assert(mHealthUser1DF.count() == 1 && mHealthUser2DF.count() == 1 && mHealthUser3DF.count() == 1)

    val mHealthJoined = mHealthUser1DF.union(mHealthUser2DF).union(mHealthUser3DF)
    mHealthJoined.show()
    assert(mHealthJoined.count() == 3)
  }

}