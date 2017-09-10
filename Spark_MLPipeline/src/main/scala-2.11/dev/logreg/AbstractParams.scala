package dev.logreg

/**
  * Created by lucieburgess on 10/09/2017.
  * @author https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/mllib/AbstractParams.scala
  * Trying to follow the Developer API example but this class doesn't seem to exist in either ML or MLlib, only in the examples
  * So copying it here to see if we can make the LR case class work. Other option is defining new parameters in a trait, like the ELM version
  */

import scala.reflect.runtime.universe._

  /**
    * Abstract class for parameter case classes.
    * This overrides the [[toString]] method to print all case class fields by name and value.
    * @tparam T  Concrete parameter class.
    */
  abstract class AbstractParams[T: TypeTag] {

    private def tag: TypeTag[T] = typeTag[T]

    /**
      * Finds all case class fields in concrete class instance, and outputs them in JSON-style format:
      * {
      *   [field name]:\t[field value]\n
      *   [field name]:\t[field value]\n
      *   ...
      * }
      */
    override def toString: String = {
      val tpe = tag.tpe
      val allAccessors = tpe.decls.collect {
        case m: MethodSymbol if m.isCaseAccessor => m
      }
      val mirror = runtimeMirror(getClass.getClassLoader)
      val instanceMirror = mirror.reflect(this)
      allAccessors.map { f =>
        val paramName = f.name.toString
        val fieldMirror = instanceMirror.reflectField(f)
        val paramValue = fieldMirror.get
        s"  $paramName:\t$paramValue"
      }.mkString("{\n", ",\n", "\n}")
    }
  }

