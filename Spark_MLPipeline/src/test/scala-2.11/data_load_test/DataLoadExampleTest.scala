package data_load_test

/**
  * Created by lucieburgess on 13/08/2017.
  * Uses person.txt in the Spark/scala/resources folder as a simple test case to test loading and basic manipulation
  * of data from a single file
  */

import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterEach, FunSuite, Matchers}

case class Person(name: String, age: Int)

class DataLoadExampleTest extends FunSuite with BeforeAndAfterEach {

  private val master = "local[*]"
  private val appName = "data_load_testing"

  var spark: SparkSession = _

  override def beforeEach() {
    spark = new SparkSession.Builder().appName(appName).master(master).getOrCreate()
  }

  test("[01] Creating Dataframe should produce Dataframe of correct size") {

    val sQLContext = spark.sqlContext
    import sQLContext.implicits._

    val df = spark.sparkContext
        .textFile("/Applications/spark-2.2.0-bin-hadoop2.7/examples/src/main/resources/people.txt")
        .map(_.split(","))
        .map(attributes => Person(attributes(0), attributes(1).trim.toInt))
        .toDF()

    assert(df.count() == 3)
    assertResult("Andy") {
      df.take(2)(1)(0) // takes the first n rows and then uses matrix notation to access the elements e.g. Andy is at position (1,0)
    }
    assertThrows[IndexOutOfBoundsException] {
      df.take(1)(1)(0) // because df.take(1) takes the first row, therefore the index of (1,0) is not accessible
    }
    assert(df.filter($"age" > 21).count() == 2)
  }

  override def afterEach() = {
    spark.stop()
  }
}


