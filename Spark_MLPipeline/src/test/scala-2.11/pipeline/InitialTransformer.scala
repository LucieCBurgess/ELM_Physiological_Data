package pipeline

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

/**
  * Created by lucieburgess on 25/08/2017.
  * Simple Transformer example for the following operations:
  * val mHealthIndexed: DataFrame = mHealthJoined.withColumn("uniqueID", monotonically_increasing_id())
  * val df2 = data
        .filter($"activityLabel" > 0)
        .withColumn("indexedLabel",when($"activityLabel".between(1, 3), 0).otherwise(1))
  */
class InitialTransformer extends Transformer {

  override val uid: String = "Binarizes abstractLabel, filters and adds monoticID"

  override def copy (extra: ParamMap): Transformer = ???

  override def transform(dataset: Dataset[_]): DataFrame = ???

  override def transformSchema(schema: StructType): StructType = ???

}
