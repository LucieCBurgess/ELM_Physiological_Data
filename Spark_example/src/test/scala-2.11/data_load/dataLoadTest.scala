package data_load

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

/**
  * Created by lucieburgess on 06/08/2017.
  * FIXME - incomplete
  */
class dataLoadTest extends FlatSpec with Matchers with BeforeAndAfter {

    private val master = "local[2]"
    private val appName = "data_load_testing"
    private var sc: SparkContext = _

    before {
      val conf = new SparkConf().setMaster(master).setAppName(appName)
    }

    "Running build()" should "create a dataframe with correct number of columns" in {

      val spark = new SparkSession.Builder().appName(appName).getOrCreate()
      case class Record(columnX: Double, columnY: Double, columnZ: Int, columnW: String)
      import spark.implicits._

      val df = spark.createDataFrame(Seq(1,2,3).map(Record(_)))
      assert(df.count()>0)

    }


    after {
      if (sc != null) {
        sc.stop()
      }
    }
}
