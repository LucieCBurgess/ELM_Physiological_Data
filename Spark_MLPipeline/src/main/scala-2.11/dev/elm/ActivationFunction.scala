package dev.elm

/**
  * Created by lucieburgess on 31/08/2017.
  */
class ActivationFunction(val af: String) {

  def function: (Double => Double) = af match {
    case "sigmoid" => ActivationFuncSigmoid.sigmoid
    case "tanh" => ActivationFuncTanh.tanh
    case "rbf" => ActivationFuncRBF.rbf
    case "step" => ActivationFuncStep.step
    case _ => throw new IllegalArgumentException("This activation function is not accepted. Please choose one of the available" +
      "activation functions")
  }
}

/** select activationFunc based on input String */
object ActivationFuncSigmoid {
  def sigmoid(x: Double): Double = 1 / (1 + math.pow(math.E, -x))
}

object ActivationFuncTanh {
  def tanh(x: Double): Double = math.tanh(x)
}

//FIXME - not yet implemented
object ActivationFuncRBF {
  def rbf(x: Double): Double = ???
}

//FIXME - not yet implemented
object ActivationFuncStep {
  def step(x: Double): Double = ???
}