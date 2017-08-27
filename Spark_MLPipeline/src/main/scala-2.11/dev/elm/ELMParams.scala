package dev.elm

/**
  * Created by lucieburgess on 27/08/2017.
  */
case class ELMParams (dataFormat: String = "text converted to DataFrame",
                          activationFunc: String = "Sigmoid",
                          hiddenNodes: Int = 10,
                          fracTest: Double = 0.5)

