# Upload data files from MHealth dataset into R and set as a data frame
setwd("/Users/lucieburgess/Documents/Birkbeck/MSc_project/MHEALTHDATASET")
WD <- getwd()

#Upload file and set column names
mHealth_1 <- read.table("mHealth_subject1.log", header = FALSE)
size <- dim(mHealth_1)
size # 161280, 24 columns
colnames(mHealth_1) <- c("Acc_Chest_X","Acc_Chest_Y","Acc_Chest_Z","ECG_1","ECG_2","Acc_LAnkle_X","Acc_LAnkle_Y","Acc_LAnkle_Z","Gyro_Ankle_X","Gyro_Ankle_Y","Gyro_Ankle_Z","Magno_Ankle_X","Magno_Ankle_Y","Magno_Ankle_Z","Acc_Arm_X","Acc_Arm_Y","Acc_Arm_Z","Gyro_Arm_X","Gyro_Arm_Y","Gyro_Arm_Z","Magno_Arm_X","Magno_Arm_Y","Magno_Arm_Z","Activity_Label")

#Observations taken at 50Hz so 50 readings per second
plot(mHealth_1$Acc_Chest_X[1:1000],type = "l")
plot(mHealth_1$Activity_Label,type = "l")

#Create a new data table that includes only the labelled data
not_labelled <- mHealth_1[mHealth_1$Activity_Label==0,]
mHealth_1_labelled <- mHealth_1[mHealth_1$Activity_Label!=0,]
dim(not_labelled) #126106, 24 columns
dim(mHealth_1_labelled) #35174, 24 columns

# Create new labels: 0 for inactive, 1 for active, and a new column 'Activity_Label_2'
# Note that inactive labels are L1: Standing still (1 min), L2: Sitting and relaxing (1 min) L3: Lying down (1 min) )

mHealth_1_labelled$Activity_Label_2 <- ifelse((mHealth_1_labelled$Activity_Label==1 | mHealth_1_labelled$Activity_Label==2 | mHealth_1_labelled$Activity_Label==3),0,1)
plot(mHealth_1_labelled$Activity_Label,type = "l")
plot(mHealth_1_labelled$Activity_Label_2,type = "l")
dim(mHealth_1_labelled) #35174 rows, 25 columns

# Add a velocity column for each set of accelerometer data - see https://physics.stackexchange.com/questions/153159/calculate-speed-from-accelerometer
# We use delta t = 0.02s as observations are sampled at 50Hz (= 50 observations per second)
# Hypothesis: the best predictor of activity is the ankle accelerometer
# Could also look at the magnetometer and gyroscope data e.g. add a tilt column if there is time, although this is not key to the computer science problem

mHealth_1_labelled$Velocity_Chest <- sqrt((mHealth_1_labelled$Acc_Chest_X*0.02)^2+(mHealth_1_labelled$Acc_Chest_Y*0.02)^2+(mHealth_1_labelled$Acc_Chest_Z*0.02)^2)
mHealth_1_labelled$Velocity_LAnkle <- sqrt((mHealth_1_labelled$Acc_LAnkle_X*0.02)^2+(mHealth_1_labelled$Acc_LAnkle_Y*0.02)^2+(mHealth_1_labelled$Acc_LAnkle_Z*0.02)^2)
mHealth_1_labelled$Velocity_Arm <- sqrt((mHealth_1_labelled$Acc_Arm_X*0.02)^2+(mHealth_1_labelled$Acc_Arm_Y*0.02)^2+(mHealth_1_labelled$Acc_Arm_Z*0.02)^2)

dim(mHealth_1_labelled) #35174 rows, 28 cols

########################################################################################################

# Try an ELM on the data and see if we can predict active or inactive using the ELMR from the data
# ELMR is an Online Sequential Extreme Learning Machine from the Cran R repository, developed in 2015
# This application can use more than one feature using y~x1+x2+x3 etc where y is the label and x is a feature
install.packages("ELMR") # see https://cran.r-project.org/web/packages/ELMR/index.html

library(ELMR)
# Example: 
x = runif(100,0,50)
y = sqrt(x)
train1 = data.frame(y,x)
dim(train1) # 100 observations of 2 variables
train1 = data.frame(preProcess(train1)) #pre-process function which scales variables between (-1,1)
OSelm_train.formula(y~x, train1, "regression", 40, "hardlim", 10, 10)

# Using the mHealth data for 1 user:
set.seed(1234)
smp_size  <- floor(0.50*nrow(mHealth_1_labelled))
train_index <- sample(seq_len(nrow(mHealth_1_labelled)),size = smp_size)

train1 <- mHealth_1_labelled[train_index,]
dim(train1) # 17587, 28
test1 <- mHealth_1_labelled[-train_index,]
dim(test1) # 17587, 28

train1 = data.frame(train1$Velocity_Chest, train1$Velocity_LAnkle, train1$Velocity_Arm, train1$Activity_Label_2)
dim(train1) # 17587, 4
train1 = data.frame(preProcess(train1))
dim(train1) #17587, 4

OSelm_train.formula(train1$train1.Activity_Label_2~train1$train1.Velocity_Chest, train1, "classification", 40, "sig", 10, 10)

install.packages("tictoc")
library(tictoc)

tic("Running the ELM using three regular features")
OSelm_train.formula(mHealth_1_labelled$Activity_Label_2~mHealth_1_labelled$Acc_Chest_X+mHealth_1_labelled$Acc_Chest_Y+mHealth_1_labelled$Acc_Chest_Z, mHealth_1_labelled,"classification", 10, "sig", 10, 10)
toc()
# This gives a poor 48.8% training accuracy
# 8.264s to train the model using 3 features and 40 hidden nodes
# 6.82s to train the model using 3 features and 10 hidden nodes

train2 = data.frame(train1$Activity_Label_2, train1$Velocity_Chest, train1$Velocity_LAnkle, train1$Velocity_Arm)
train2 = data.frame(preProcess(train2))
dim(train2) # 17587 columns, 4 rows

tic("Running the ELM using three velocity features")
OSelm_train.formula(train2$Activity_Label_2~train2$Velocity_Chest+train2$Velocity_LAnkle+train2$Velocity_Arm, train2,"classification", 40, "sig", 10, 10)
toc()
# Without pre-processing:
# 74.4% accuracy, 6.725s to train the model using 3 features and 10 hidden nodes
# 73.3% accuracy, 8.122s to train the model using 3 features and 40 hidden nodes



########################################################################################
install.packages("elmNN") # see
library(elmNN)

set.seed(1234)
smp_size  <- floor(0.50*nrow(mHealth_1_labelled))
train_index <- sample(seq_len(nrow(mHealth_1_labelled)),size = smp_size)

train <- mHealth_1_labelled[train_index,]
test <- mHealth_1_labelled[-train_index,]

model <- elmtrain(x=train$Velocity_LAnkle, y=train$Activity_Label_2, nhid=100, actfun="sig")
prediction <- predict(model,newdata=test$Velocity_LAnkle)

test <- cbind(test,prediction)
View(test)
print(model)

#Prediction values are a probability so we need to convert them to (0,1) in order to calculate the confusion matrix
#Calculate the test MSE
# Use a for loop to calculate the optimal value for the fitted(x) for which we choose (0 | 1)

testMSE<-rep(0,20)
for(i in 1:20) {
  test$Binomial_prediction <- ifelse(test$prediction<=(0.05*i),0,1)
  testMSE[i] <- mean(test$Binomial_prediction!=test$Activity_Label_2)
}
testMSE

# 0.6 is the threshold which minimises the error based on fitted(x) -> (0 | 1)
# This gives an error rate of 26.7%, which seems reasonable for one feature for this classifier
# testMSE_final is the error rate for a threshold vector value of 0.6 for 'active' vs 'inactive'

test$Binomial_prediction <- ifelse(test$prediction<=0.6,0,1)
testMSE_final <- mean(test$Binomial_prediction!=test$Activity_Label_2)
testMSE_final

# Using a probability threshold of 0.6, tune for a different activation function. First of all try hardlimit
library(elmNN)
set.seed(2)
smp_size  <- floor(0.75*nrow(mHealth_1_labelled))
train_index <- sample(seq_len(nrow(mHealth_1_labelled)),size = smp_size)

train <- mHealth_1_labelled[train_index,]
test <- mHealth_1_labelled[-train_index,]

model <- elmtrain(x=train$Velocity_LAnkle, y=train$Activity_Label_2, nhid=100, actfun="hardlim")
prediction_hl <- predict(model,newdata=test$Velocity_LAnkle)

test <- cbind(test,prediction_hl)
print(model)

testMSE<-rep(0,20)
for(i in 1:20) {
  test$Binomial_prediction <- ifelse(test$prediction_hl<=(0.05*i),0,1)
  testMSE[i] <- mean(test$Binomial_prediction!=test$Activity_Label_2)
}
testMSE

# Try a radial basis function

model <- elmtrain(x=train$Velocity_LAnkle, y=train$Activity_Label_2, nhid=100, actfun="radbas")
prediction_rb <- predict(model,newdata=test$Velocity_LAnkle)

test <- cbind(test,prediction_rb)
print(model)

testMSE<-rep(0,20)
for(i in 1:20) {
  test$Binomial_prediction <- ifelse(test$prediction_rb<=(0.05*i),0,1)
  testMSE[i] <- mean(test$Binomial_prediction!=test$Activity_Label_2)
}
testMSE

# Different activation functions seem to give result of approx 26% error rate for a value of 0.6 probability for (0 | 1)
# Hardlimit function seems to give slightly improved error rate

# Now tune for the number of hidden neurons
# This code falls over DO NOT RUN
# There is no error in the code, it's just too slow
#########################################################################

testMSE <- rep(0,12)
hidden_neurons = c(1,5,10,50,100,200,500,1000)
for (i in hidden_neurons) {
  model <- elmtrain(x=train$Velocity_LAnkle, y=train$Activity_Label_2, nhid=hidden_neurons[i], actfun="hardlim")
  prediction_hl <- predict(model,newdata=test$Velocity_LAnkle)
  test <- cbind(test,prediction_hl)
  print(model)
  test$Binomial_prediction <- ifelse(test$prediction_hl<=0.6,0,1)
  testMSE[i] <- mean(test$Binomial_prediction!=test$Activity_Label_2)
}
testMSE
############################################################################

# 500 hidden neurons takes about 30 seconds, 1,000 hidden neurons takes about 3 minutes
print(system.time(model <- elmtrain(x=train$Velocity_LAnkle, y=train$Activity_Label_2, nhid=500, actfun="hardlim")))
prediction_hl <- predict(model,newdata=test$Velocity_LAnkle)
test <- cbind(test,prediction_hl)
print(model)

# For 1,000 hidden neurons, find the value of test$prediction_hl that gives the lowest test MSE for a value of (0 | 1)
testMSE<-rep(0,20)
for(i in 1:20) {
  test$Binomial_prediction <- ifelse(test$prediction_hl<=(0.05*i),0,1)
  testMSE[i] <- mean(test$Binomial_prediction!=test$Activity_Label_2)
}
testMSE

# Now trying a logistic binomial regression based on a number of factors, to understand which is the best predictor of activity level

#glm.fit=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Weekly,family=binomial)
glm.fit=glm(Activity_Label_2~Acc_Chest_X+Acc_Chest_Y+Acc_Chest_Z+ECG_1+ECG_2+Acc_LAnkle_X+Acc_LAnkle_Y+Acc_LAnkle_Z+Gyro_Ankle_X+Gyro_Ankle_Y+Gyro_Ankle_Z+Magno_Ankle_X+Magno_Ankle_Y+Magno_Ankle_Z+Acc_Arm_X+Acc_Arm_Y+Acc_Arm_Z+Gyro_Arm_X+Gyro_Arm_Y+Gyro_Arm_Z+Magno_Arm_X+Magno_Arm_Y+Magno_Arm_Z+Activity_Label, data=train, family=binomial)
summary(glm.fit)
glm.probs.train=predict(glm.fit,data=train,type='response')
glm.probs.train[1:10]
glm.predict.train=rep(0,26380)
glm.predict.train[glm.probs.train>0.4]=1
View(glm.predict.train)

train <- cbind(train,glm.predict.train)
View(train)

train_MSE <- mean(train$glm.predict.train!=train$Activity_Label_2)
train_MSE # 28.97% error

# Now trying a logistic binomial regression based on a number of factors, to understand which is the best predictor of activity level

#glm.fit=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Weekly,family=binomial)
glm.fit=glm(Activity_Label_2~Velocity_LAnkle+Velocity_Chest+Velocity_Arm+ECG_1+ECG_2,data=train, family=binomial)
summary(glm.fit)
glm.probs.train=predict(glm.fit,data=train,type='response')
glm.probs.train[1:10]
glm.predict.train=rep(0,26380)
glm.predict.train[glm.probs.train>0.4]=1
View(glm.predict.train)

train <- cbind(train,glm.predict.train)
View(train)

train_MSE <- mean(train$glm.predict.train!=train$Activity_Label_2)
train_MSE # 28.97% error

# Calculate HRV (Heart-Rate-Variability) instead of ECG signal, which is not very significant predictor according to the glm
plot(mHealth_1$ECG_1[2000:2200],type = "l")





  



