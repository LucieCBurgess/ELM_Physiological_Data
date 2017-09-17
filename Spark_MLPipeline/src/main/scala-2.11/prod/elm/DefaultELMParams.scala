package prod.elm

/**
  * Created by lucieburgess on 27/08/2017.
  * Sets default ELM params, for use in running the model and command line options parsing.
  */
case class DefaultELMParams(activationFunc: String = "sigmoid",
                            hiddenNodes: Int = 50,
                            fracTest: Double = 0.5)

