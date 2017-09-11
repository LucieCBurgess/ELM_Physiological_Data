name := "Spark_MLPipeline"

version := "1.0"

scalaVersion := "2.11.8"

resolvers += "Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/"

// Scalatest dependencies
libraryDependencies += "org.scalactic" %% "scalactic" % "3.0.1"

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.1" % "test"

// Spark dependencies
lazy val sparkVersion = "2.2.0"

libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion

libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion

libraryDependencies += "org.apache.spark" %% "spark-streaming" % sparkVersion

libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion

// Scopt dependencies
libraryDependencies += "com.github.scopt" %% "scopt" % "3.6.0"

// Breeze dependencies https://github.com/scalanlp/breeze/wiki/Installation
libraryDependencies  ++= Seq(
  "org.scalanlp" %% "breeze" % "0.13.2",
  "org.scalanlp" %% "breeze-natives" % "0.13.2",
  // the visualization library is distributed separately as well. It depends on LGPL code
  "org.scalanlp" %% "breeze-viz" % "0.13.2",
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
)

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"



    