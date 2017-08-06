package data_load

import org.apache.spark.sql.SparkSession

/**
  * Created by lucieburgess on 04/08/2017.
  * Creates a Spark dataframe for the mHealth dataset for user1.
  * All data in the mHealth dataset are doubles apart from activity_Label which is an integer
  * We will create a separate case class for each subject, create a data frame for each,
  * and then combine the dataframes to get a single dataframe for all users
  * Code will be tested in a separate package of the same name in test
  *
  * From: https://spark.apache.org/docs/latest/sql-programming-guide.html#getting-started
  * A dataset is a distributed collection of data. Dataset is a new interface added in Spark 1.6. The API is available in Scala and Java
  * A Dataframe is a dataset organised into named columns. A DataFrame is represented by a Dataset of Rows.
  * In the Scala API, DataFrame is simply a type alias of Dataset[Row].
  *
  * From https://spark.apache.org/docs/latest/sql-programming-guide.html#data-sources
  * The Scala interface for Spark SQL supports automatically converting an RDD containing case classes to a DataFrame
  * The case class defines the scheme of the table
  * The names of the arguments to the case classes are read using reflection and become the names of the columns
  * So we define the mHealth_subject1.txt file schema using a case class and infer it using reflection, as shown below
  */

// FIXME - not completed
// FIXME - need to add a subjectID and a rowID to the record
// FIXME - not unit tested
// FIXME - configurations need to be set at some point

case class mHealthUser1(acc_Chest_X: Double, acc_Chest_Y: Double, acc_Chest_Z: Double,
                        ecg_1: Double, ecg_2: Double,
                        acc_Ankle_X: Double, acc_Ankle_Y: Double, acc_Ankle_Z: Double,
                        gyro_Ankle_X: Double, gyro_Ankle_Y: Double, gyro_Ankle_Z: Double,
                        magno_Ankle_X: Double, magno_Ankle_Y: Double, magno_Ankle_Z: Double,
                        acc_Arm_X: Double, acc_Arm_Y: Double, acc_Arm_Z: Double,
                        gyro_Arm_X: Double, gyro_Arm_Y: Double, gyro_Arm_Z: Double,
                        magno_Arm_X: Double, magno_Arm_Y: Double, magno_Arm_Z: Double,
                        activityLabel: Int)

object dataLoad {

  val spark = SparkSession
    .builder()
    .appName("Loading data")
    .config("spark.some.config.option", "some-value")
    .getOrCreate()

  def build() :Unit = {

    import spark.implicits._

    val mHealthUser1DF = spark.sparkContext
      .textFile("/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET/mHealth_subject1.txt")
      .map(_.split(" "))
      .map(attributes => mHealthUser1(attributes(0).toDouble, attributes(1).toDouble, attributes(3).toDouble, attributes(4).toDouble,
        attributes(5).toDouble, attributes(6).toDouble, attributes(7).toDouble, attributes(8).toDouble,
        attributes(9).toDouble, attributes(10).toDouble, attributes(11).toDouble, attributes(12).toDouble, attributes(13).toDouble,
        attributes(14).toDouble, attributes(15).toDouble, attributes(16).toDouble, attributes(17).toDouble, attributes(18).toDouble,
        attributes(19).toDouble, attributes(20).toDouble, attributes(21).toDouble, attributes(22).toDouble, attributes(23).toDouble,
        attributes(24).toInt))
      .toDF()

    }
}




// def main(args: Array[String]): Unit = {
// val conf = new SparkConf() // defining Spark configuration object
// .setMaster("local[*]") // set SparkMaster as local
// .setAppName("Loading data")
// .set("spark.executor.memory", "4g")

// val sc = new SparkContext(conf)

// sc.textFile method creates a distributed RDD
// val mHealthUser1File = sc.textFile("/Users/lucieburgess/Documents/Birkbeck/MSc_Project/MHEALTHDATASET")

// val mHealthUser1FileNext = mHealthUser1File.map(x => x.split(" ")).map {
//   case Array(acc_Chest_X, acc_Chest_Y, acc_Chest_Z, ecg_1, ecg_2,
//   acc_LAnkle_X, acc_LAnkle_Y, acc_LAnkle_Z, gyro_Ankle_X, gyro_Ankle_Y, gyro_Ankle_Z,
//   magno_Ankle_X, magno_Ankle_Y, magno_Ankle_Z, acc_Arm_X, acc_Arm_Y, acc_Arm_Z,
//   gyro_Arm_X, gyro_Arm_Y, gyro_Arm_Z, magno_Arm_X, magno_Arm_Y, magno_Arm_Z, activityLabel) =>
//   mHealthUser1(acc_Chest_X.toDouble, acc_Chest_Y.toDouble, acc_Chest_Z.toDouble, ecg_1.toDouble, ecg_2.toDouble,
//   acc_LAnkle_X.toDouble, acc_LAnkle_Y.toDouble, acc_LAnkle_Z.toDouble, gyro_Ankle_X.toDouble, gyro_Ankle_Y.toDouble, gyro_Ankle_Z.toDouble,
//  magno_Ankle_X.toDouble, magno_Ankle_Y.toDouble, magno_Ankle_Z.toDouble, acc_Arm_X.toDouble, acc_Arm_Y.toDouble, acc_Arm_Z.toDouble,
//  gyro_Arm_X.toDouble, gyro_Arm_Y.toDouble, gyro_Arm_Z.toDouble, magno_Arm_X.toDouble, magno_Arm_Y.toDouble, magno_Arm_Z.toDouble,
//   activityLabel.toInt)

// mHealthUser1FileNext.toDF()









