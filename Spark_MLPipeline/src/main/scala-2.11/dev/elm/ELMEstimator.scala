package dev.elm

import org.apache.spark.ml.classification.Classifier
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * Created by lucieburgess on 27/08/2017.
  */
class ELMEstimator(val uid: String) extends Classifier {

  def this() = this(Identifiable.randomUID("ELM proof-of-concept"))

  override def copy(extra: ParamMap) = ???

  override def train(dataset: Dataset[_]) = ???

}
