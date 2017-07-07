# Upload data files from MHealth dataset into R and set as a data frame
setwd("/Users/lucieburgess/Documents/Birkbeck/MSc_project/MHEALTHDATASET")
WD <- getwd()
WD

#Upload file and set column names

mHealth_1 <- read.table("mHealth_subject1.log", header = FALSE)
colnames(mHealth_1) <- c("Acc_Chest_X","Acc_Chest_Y","Acc_Chest_Z","ECG_1","ECG_2","Acc_LAnkle_X","Acc_LAnkle_Y","Acc_LAnkle_Z","Gyro_Ankle_X","Gyro_Ankle_Y","Gyro_Ankle_Z","Magno_Ankle_X","Magno_Ankle_Y","Magno_Ankle_Z","Acc_Arm_X","Acc_Arm_Y","Acc_Arm_Z","Gyro_Arm_X","Gyro_Arm_Y","Gyro_Arm_Z","Magno_Arm_X","Magno_Arm_Y","Magno_Arm_Z","Activity_Label")
#Observations taken at 50Hz so 50 readings per second
plot(mHealth_1$Acc_Chest_X[1:1000],type = "l")
plot(mHealth_1$Activity_Label,type = "l")

#Create a new data table that includes only the labelled data

not_labelled <- mHealth_1[mHealth_1$Activity_Label==0,"id"]
mHealth_1_labelled <- subset(mHealth_1, !(id %in% not_labelled))

#Create new labels: 0 for inactive, 1 for active
# Note that inactive labels are L1: Standing still (1 min), L2: Sitting and relaxing (1 min) L3: Lying down (1 min) )

mHealth_1_labelled$Activity_Label_2 <- ifelse((mHealth_1$Activity_Label==1 || mHealth_1$Activity_Label==2 || mHealth_1$Activity_Label==3),0,1)

# Try an ELM on the data and see if we can predict active or inactive using the ELMR from the data
install.packages("ELMR")
  



