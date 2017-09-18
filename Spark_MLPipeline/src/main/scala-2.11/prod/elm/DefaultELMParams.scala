package prod.elm

/**
  * Created by lucieburgess on 27/08/2017.
  * Sets default ELM params, for use in running the model and command line options parsing.
  */
case class DefaultELMParams(activationFunc: String = "tanh",
                            hiddenNodes: Int = 100,
                            fracTest: Double = 0.6)

