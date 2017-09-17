package prod.elm

import org.apache.log4j.{Level, Logger}

/**
  * Created by lucieburgess on 27/08/2017.
  * Defines a number of parameters for this ELM and incorporates main method to run ELM
  * Set the layout for using the parsing the parameters using Scopt, https://github.com/scopt/
  */

object ELMMain {

  /** switches off verbose output to the console. Can be set to INFO if preferred */
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)

  def main(args: Array[String]): Unit = {

    val defaultParams = DefaultELMParams()

    val parser = new scopt.OptionParser[DefaultELMParams]("Extreme Learning Machine parameters for class ELM") {

      head("ELM parameters")

      opt[String]("activationFunc").text(s"Sets the Activation Function which modifies the hidden layer output matrix, default: ${defaultParams.activationFunc}")
        .action((x, c) => c.copy(activationFunc = x))

      opt[Int]("hiddenNodes").text(s"Sets the number of hidden nodes in the hidden layer, default: ${defaultParams.hiddenNodes}")
        .action((x, c) => c.copy(hiddenNodes = x))

      opt[Double]("fracTest").text(s"The fraction of the data to use for testing, default: ${defaultParams.fracTest}")
        .action((x, c) => c.copy(fracTest = x))

      checkConfig { params =>
        if (params.fracTest < 0.1 || params.fracTest >= 0.6) {
          failure(s"fracTest ${params.fracTest} value is incorrect; it should be in range [0,1).")
        } else {
          success
        }
      }
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => ELMPipeline.run(params)
      case _ => sys.exit(1)
    }
  }
}
