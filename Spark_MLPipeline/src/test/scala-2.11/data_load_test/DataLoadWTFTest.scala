package data_load_test

import dev.data_load.{DataLoadOption, DataLoadWTF}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.scalatest.FunSuite

/**
  * @author lucieburgess on 10/09/2017.
  * Tests for loading multiple files using spark.sparkContext.wholeTextFile method.
  * Tests for 3 and 10 files.
  * ALL TESTS PASS
  */
class DataLoadWTFTest extends FunSuite with SparkSessionTestWrapper {


    test("[01] Loading data from three files results in single file of correct size") {

      val folderNameTest1: String = "Multiple3"

      val dfs: Seq[DataFrame] = Seq("mHealth_subject1.txt","mHealth_subject2.txt","mHealth_subject3.txt").flatMap(DataLoadOption.createDataFrame)

      val size1 = dfs.head.count()
      val size2 = dfs(1).count()
      val size3 = dfs(2).count()

      val data = DataLoadWTF.createDataFrame(folderNameTest1)
      assert(data.count == (size1 + size2 + size3))
    }

    test("[02] Calling wholeTextFile creates RDD Map") {

      val folderNameTest2: String = "Multiple3"

      val data: RDD[(String, String)] = spark.sparkContext.wholeTextFiles("/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/" + folderNameTest2, 3)

      assert(data.isInstanceOf[RDD[(String,String)]])

      val fileNames: Array[String] = data.map{case (fileName, content) => fileName}.collect()

      assert(fileNames.length == 3)

      println(fileNames.mkString(","))

      assertResult("file:/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/Multiple3/mHealth_subject1.txt"){
        fileNames.head }
      assertResult("file:/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/Multiple3/mHealth_subject2.txt"){
        fileNames(1) }
      assertResult("file:/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/Multiple3/mHealth_subject3.txt"){
        fileNames(2) }

    }

    test("[03] Loading data from all 10 files results in single file of correct size") {

      val start = System.nanoTime()

      val folderNameTest3: String = "Multiple10"

      val largeFiles: Seq[String] = Seq("mHealth_subject1.txt", "mHealth_subject2.txt", "mHealth_subject3.txt", "mHealth_subject4.txt",
        "mHealth_subject5.txt", "mHealth_subject6.txt", "mHealth_subject7.txt", "mHealth_subject8.txt",
        "mHealth_subject9.txt", "mHealth_subject10.txt")


      val dfs: Seq[DataFrame] = largeFiles.flatMap(DataLoadOption.createDataFrame)

      val data = DataLoadWTF.createDataFrame(folderNameTest3)

      val finish = (System.nanoTime() - start) / 1e9

      println(s"Time to construct complete DataFrame: $finish seconds")

      assert(data.count == dfs.map(d => d.count).reduce(_+_))
    }

}
