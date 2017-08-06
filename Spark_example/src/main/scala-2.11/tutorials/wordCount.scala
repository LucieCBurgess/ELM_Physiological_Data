package tutorials

/**
  * Created by lucieburgess on 03/08/2017.
  */

import org.apache.spark.{SparkConf, SparkContext}

object wordCount {

  def main(args: Array[String]): Unit = {
     val conf = new SparkConf() // defining Spark configuration object
        .setMaster("local[*]") // set SparkMaster as local
        .setAppName("Spark test")
        .set("spark.executor.memory", "4g")

    val sc = new SparkContext(conf)

    val t1 = System.nanoTime()

    // Defines the parallel action on the spark context
    val lines = sc.parallelize(Seq("This is the first line", "This is the second line", "This is the third line"))

    val counts = lines.flatMap(line => line.split(" "))
                      .map(word => (word, 1))
                      .reduceByKey(_+_)
      counts.foreach(println)

    val duration = (System.nanoTime() - t1)/ 1e9d

    println("****************"+duration+"******************")
  }
}
