package tutorials


/**
  * Created by lucieburgess on 03/08/2017.
  */
object wordCountNotParallel {

    def reduceByKey[K,V](collection: Traversable[Tuple2[K,V]])(implicit num: Numeric[V]) = {
      import num._
      collection
        .groupBy(_._1)
        .map { case (group: K, traversable) => traversable.reduce { (a, b) => (a._1, a._2 + b._2) }
        }
    }

    def main(args: Array[String]): Unit = {

      // Defines the parallel action on the spark context
      val lines = Seq("This is the first line", "This is the second line", "This is the third line")


      val counts = lines.flatMap(line => line.split(" ")).map(word => (word, 1))
      reduceByKey(counts)
      counts.foreach(println)
    }
}
