package pipeline

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.BinaryAttribute
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._

/**
  * Created by lucieburgess on 25/08/2017.
  * Simple Transformer example for the following operations:
  * val mHealthIndexed: DataFrame = mHealthJoined.withColumn("uniqueID", monotonically_increasing_id())
  * val df2 = data
        .filter($"activityLabel" > 0)
        .withColumn("indexedLabel",when($"activityLabel".between(1, 3), 0).otherwise(1))
  */
class MyBinarizer(override val uid: String) extends Transformer {

  def this() = this(Identifiable.randomUID("initialTransformer"))

  override def copy (extra: ParamMap): MyBinarizer = defaultCopy(extra)

  val inputColName = "activityLabel"
  val outputColName = "binaryLabel"

  override def transform(dataset: Dataset[_]): DataFrame = {
    val outputSchema = transformSchema(dataset.schema, logging = true)
    val schema = dataset.schema
    val inputType = schema("activityLabel").dataType
    val threshold:Int = 3

    val binarizerInt = udf { in: Int => if (in > threshold) 1.0 else 0.0}

    val metadata = outputSchema(outputColName).metadata

    inputType match {
      case IntegerType => dataset.select(col("*"),binarizerInt(col(inputColName)).as(outputColName, metadata))
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    // input col is $activityLabel, output col is $binaryLabel

    val inputType = schema(inputColName).dataType

    val outputCol: StructField = inputType match {
      case IntegerType => BinaryAttribute.defaultAttr.withName(outputColName).toStructField()
      case _ => throw new IllegalArgumentException(s"Data type $inputType is not supported")
    }

    if (schema.fieldNames.contains(outputColName)) {
      throw new IllegalArgumentException(s"Output column $outputColName already exists")
    }
    StructType(schema.fields :+ outputCol)
  }

}
