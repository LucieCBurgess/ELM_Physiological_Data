package elm_test

import data_load_test.SparkSessionTestWrapper
import dev.elm.ELMParamsMine
import org.scalatest.FunSuite

/**
  * Created by lucieburgess on 27/08/2017.
  */
class ELMParamsMineTest extends FunSuite with SparkSessionTestWrapper {

  //override val activationFuncs: Array[String] = Array("Sigmoid","RBF", "Tanh","Step")

  val afParams = new ELMParamsMine()

  test("[09] getActivationFunction returns correct String") {

    val result = afParams.getActivationFunc("Sigmoid")
    assertResult("Sigmoid")

    val result2 = afParams.getActivationFunc("Hyperbolic")
    assertThrows(IllegalArgumentException)

  }
}
