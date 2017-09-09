## Initial steps to get SparkR working natively with R

# Trying Spark with R to see if we can get an increase in the speed
Sys.getenv()

# This from the SparkR tutorial
install.packages("sparkR") # gives an error

if (nchar(Sys.getenv("SPARK_HOME")) < 1) {
  Sys.setenv(SPARK_HOME = "/Applications/spark-2.2.0-bin-hadoop2.7")
}
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "2g"))

install.packages("SparkR")
library(SparkR)

# Load the mHealth data into R
setwd("/Users/lucieburgess/Documents/Birkbeck/MSc_project/MHEALTHDATASET")
WD <- getwd()
mHealth_1 <- read.table("mHealth_subject1.log", header = FALSE)
colnames(mHealth_1) <- c("Acc_Chest_X","Acc_Chest_Y","Acc_Chest_Z","ECG_1","ECG_2","Acc_LAnkle_X","Acc_LAnkle_Y","Acc_LAnkle_Z","Gyro_Ankle_X","Gyro_Ankle_Y","Gyro_Ankle_Z","Magno_Ankle_X","Magno_Ankle_Y","Magno_Ankle_Z","Acc_Arm_X","Acc_Arm_Y","Acc_Arm_Z","Gyro_Arm_X","Gyro_Arm_Y","Gyro_Arm_Z","Magno_Arm_X","Magno_Arm_Y","Magno_Arm_Z","Activity_Label")

# Create a Spark dataframe from the mHealth data
mHealth_1_sc <- as.DataFrame(mHealth_1)
head(mHealth_1_sc)

# Creating a SparkR dataframe
mHealth_1_sc <- as.DataFrame(mHealth_1_labelled) # Need to load up mHealth_1_labelled again

# Get basic information about the dataframe
mHealth_1_sc

# Filter the dataframe for only labelled data
# Count the number of rows for each type of activity label
mh <- filter(mHealth_1_sc, mHealth_1_sc$Activity_Label > 0)
head(mh)
head(summarize(groupBy(mh, mh$Activity_Label), count = n(mh$Activity_Label)), num=12L)

# Realised that this is not the way to go about it ... no way of accessing Spark's in-built parallelisation like this
# Back to Scala!

#############################################################################################
# Playing with SparklyR
# Install SparklyR and Spark. SparklyR is a package which gives a front-end for Spark using R.
# Also download a local version of Spark (in case of not using a remote cluster)

install.packages("sparklyr")
install.packages("dplyr")
library(sparklyr)
spark_available_versions()
# sparklyR seems to work with Spark v.1.6.2
spark_install(version = "1.6.2")
spark_install(version = "2.0.2", hadoop_version = "2.7")
sc <- spark_connect(master = "local")

library(dplyr)
mHealth_1_tbl <- copy_to(sc, mHealth_1_labelled)
src_tbls(sc)

mHealth_1_tbl %>%dplyr::filter(Activity_Label == 12)

