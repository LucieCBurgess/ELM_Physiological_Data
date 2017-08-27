package dev.elm

import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable

/**
  * Created by lucieburgess on 27/08/2017.
  * See https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/ml/DeveloperApiExample.scala
  */
trait ELMParamsMine extends Params {

  //def this() = this(Identifiable.randomUID("ELM Params"))

  val activationFuncs: Array[String] = Array("Sigmoid","RBF", "Tanh","Step")
  val activationFuncParam: StringArrayParam = new StringArrayParam(this,"activationFunc", s"Available activation functions: ${activationFuncs}.toString")

  activationFuncParam.->(activationFuncs)

//  def getActivationFunc(af: String):String = for (a <- activationFuncs) {
//    af match {
//      case af if af.equals(a) => af
//      case _ => throw new IllegalArgumentException("The activation function you were looking for does not exist")
//  }

  def copy(extra: ParamMap) = defaultCopy(extra)
}
