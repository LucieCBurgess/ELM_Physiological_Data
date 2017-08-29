package dev.elm

import dev.pipeline.LRTestMain.spark
import dev.pipeline.{LRPipeline, LRTestParams, SparkSessionWrapper}

/**
  * Created by lucieburgess on 27/08/2017.
  * Defines a number of parameters for this ELM and incorporates main method to run ELM
  * Set the layout for using the parsing the parameters using Scopt, https://github.com/scopt/scopt
  */

object ELMMain extends SparkSessionWrapper {


  def main(args: Array[String]): Unit = {

    val defaultParams = DefaultELMParams()

    val parser = new scopt.OptionParser[DefaultELMParams]("Extreme Learning Machine parameters for class ELM") {

      head("ELM parameters")

      opt[String]("dataFormat").text(s"Please enter the dataformat here, default: ${defaultParams.dataFormat}")
        .action((x, c) => c.copy(dataFormat = x))

      opt[String]("Activation function").text(s"Sets the Activation Function which modifies the hidden layer output matrix, default: ${defaultParams.activationFunc}")
        .action((x, c) => c.copy(activationFunc = x))

      opt[Int]("hiddenNodes").text(s"Sets the number of hidden nodes in the hidden layer, default: ${defaultParams.hiddenNodes}")
        .action((x, c) => c.copy(hiddenNodes = x))

      opt[Double]("fracTest").text(s"The fraction of the data to use for testing, default: ${defaultParams.fracTest}")
        .action((x, c) => c.copy(fracTest = x))

      checkConfig { params =>
        if (params.fracTest < 0 || params.fracTest >= 1) {
          failure(s"fracTest ${params.fracTest} value is incorrect; it should be in range [0,1).")
        } else {
          success
        }
      }

    }

    //FIXME - implement ELMpipeline once Estimator has been calculated
    //FIXME - perhaps define a trait for the pipeline so that other models can be easily implemented, e.g. MultiLayer perceptron
    parser.parse(args, defaultParams) match {
      //case Some(params) => ELMPipeline.run(params) - not yet implemented
      case Some(params) => println(s"ELM params \n$params")
      case _ => sys.exit(1)
    }
  }
  spark.stop()
}
