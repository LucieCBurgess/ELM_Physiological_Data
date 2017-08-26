package data_load_test

import org.apache.spark.sql.SparkSession

/**
  * Created by lucieburgess on 14/08/2017.
  * Alternative way to run the Spark Session in a shared environment for each test
  * see https://stackoverflow.com/questions/43729262/how-to-write-unit-tests-in-spark-2-0
  */
trait SparkSessionTestWrapper {

  lazy val spark: SparkSession = {
    SparkSession.builder().master("local[*]").appName("testing").getOrCreate()
  }

}
