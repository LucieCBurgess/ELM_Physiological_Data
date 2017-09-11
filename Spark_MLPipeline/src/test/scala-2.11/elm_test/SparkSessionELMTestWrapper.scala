package elm_test

import org.apache.spark.sql.SparkSession

/**
  * Created by lucieburgess on 11/09/2017. Separate SparkSession test wrapper for each test package to avoid
  * SparkContext closing down unnecessarily.
  */
trait SparkSessionELMTestWrapper {

  val spark: SparkSession = {
    SparkSession.builder().master("local[*]").appName("ELM_tests").getOrCreate()
  }

}
