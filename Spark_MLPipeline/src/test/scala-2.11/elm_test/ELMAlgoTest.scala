package elm_test

import data_load_test.SparkSessionTestWrapper
import dev.data_load.DataLoad
import dev.elm.{ELMClassifier, ELMModel}
import dev.elm.ELMPipeline.spark
import org.apache.spark.ml.classification.Classifier
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.{Estimator, Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorSlicer}
import org.apache.spark.sql.functions.{col, monotonically_increasing_id, when}
import org.apache.spark.sql.types.{DoubleType, StructField}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.scalatest.FunSuite

import scala.collection.mutable

/**
  * Created by lucieburgess on 02/09/2017.
  */
class ELMAlgoTest extends FunSuite with SparkSessionTestWrapper {

  import spark.implicits._

  val fileName: String = "mHealth_subject1.txt"

  /** Load training and test data and cache it */
  val data = DataLoad.createDataFrame(fileName) match {
    case Some(df) => df
      .filter($"activityLabel" > 0)
      .withColumn("binaryLabel", when($"activityLabel".between(1, 3), 0).otherwise(1))
      .withColumn("uniqueID", monotonically_increasing_id())
    case None => throw new UnsupportedOperationException("Couldn't create DataFrame")
  }

  val datasetSize: Int = data.count().toInt
  val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z")
  val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
  val preparedData: DataFrame = featureAssembler.transform(data)

  test("[01 Vector assembler adds output column of features and can be added to pipeline") {

    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z", "acc_Arm_X", "acc_Arm_Y", "acc_Arm_Z")
    val featureAssembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val preparedData = featureAssembler.transform(data)
    assert(preparedData.select("features").head.get(0).asInstanceOf[Vector].size == 6)

    preparedData.printSchema()
    val pipelineStages = new mutable.ArrayBuffer[PipelineStage]()
    pipelineStages += featureAssembler

    assert(pipelineStages.head == featureAssembler)

  }

  test("[02] Can create an array of doubles from a single DataFrame column") {

    val array: Array[Double] = preparedData.select("acc_Chest_X").as[Double].collect()
    assertResult(array.length) {
      preparedData.count().toInt
    }
  }

  test("[03] Can create an array of all the data in a features Vector to a double") {

    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z", "acc_Arm_X", "acc_Arm_Y", "acc_Arm_Z")
    val featureColsIndex = featureCols.map(c => s"${c}_index")

    val indexers = featureCols.map(
      c => new StringIndexer().setInputCol(c).setOutputCol(s"${c}_index")
    )

    val assembler = new VectorAssembler().setInputCols(featureColsIndex).setOutputCol("features")

    val slicer = new VectorSlicer().setInputCol("features").setOutputCol("double_features").setNames(featureColsIndex.init)

    val transformed = new Pipeline().setStages(indexers :+ assembler :+ slicer)
      .fit(data)
      .transform(data)

    transformed.show()
  }

  test("[04] Can create an array of all the data in a features Vector") {

    //val headers = data.schema.collect{ case StructField(name, DoubleType, nullable, meta) => name}
    // Gives an error: cannot cast Vector to Double type. The sliced features are being returned as Vectors.

    val featureCols = Array("acc_Chest_X", "acc_Chest_Y", "acc_Chest_Z")
    //val featureColsIndex = featureCols.map(c => s"${c}_index")

    val featureSlicers: Seq[VectorSlicer] = featureCols.map {
      col => new VectorSlicer().setInputCol("features").setOutputCol(s"${col}_sliced").setNames(Array(s"${col}"))
    }

    //val output = featureSlicers.map(f => f.transform(preparedData).select(f.getOutputCol).as[Double].collect)

    val output2: Seq[Vector] = featureSlicers.map(f => f.transform(preparedData).select(f.getOutputCol).asInstanceOf[Vector])

    val output3 = output2.flatMap(v => v.toArray)

    //val array = output.flatten.toArray
    println(s"************** ${output3.length} *************")

  }
}


//  This doesn't work, gives a casting error
//    val features: Vector = preparedData.select("features").asInstanceOf[Vector]
//    val array = features.toArray
//    println(array.toString)}