package data_load

//import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession.Builder
import org.scalatest.{BeforeAndAfterEach, FlatSpec, FunSuite, Matchers}

/**
  * Created by lucieburgess on 06/08/2017.
  * FIXME - incomplete
  */

case class Person(name: String, age: Int)

class dataLoadTest extends FunSuite with Matchers with BeforeAndAfterEach {

  private val master = "local[*]"
  private val appName = "data_load_testing"

  var spark: SparkSession = _

  override def beforeEach() {
    spark = new SparkSession.Builder().appName(appName).master(master).getOrCreate()
  }

  test("Creating dataframe should produce DataFrame of correct size") {

    val sQLContext = spark.sqlContext
    import sQLContext.implicits._

    val df = spark.sparkContext
      .textFile("/Applications/spark-2.2.0-bin-hadoop2.7/examples/src/main/resources/people.txt")
      .map(_.split(","))
      .map(attributes => Person(attributes(0), attributes(1).trim.toInt))
      .toDF()

      assert(df.count() == 3)
      assert(df.take(1)(0)(0).equals("Michael"))
    }

  override def afterEach() = {
    spark.stop()
  }
}
