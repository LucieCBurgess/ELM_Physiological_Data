package dev.elm

/**
  * Created by lucieburgess on 27/08/2017.
  */
case class DefaultELMParams(dataFormat: String = "text converted to DataFrame",
                            val activationFunc: String = "Sigmoid",
                            val hiddenNodes: Int = 10,
                            val fracTest: Double = 0.5)

