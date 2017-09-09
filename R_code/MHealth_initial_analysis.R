# Upload data files from MHealth dataset into R and set as a data frame
setwd("/Users/lucieburgess/Documents/Birkbeck/MSc_project/MHEALTHDATASET")
WD <- getwd()

#Upload file and set column names
mHealth_1 <- read.table("mHealth_subject1.log", header = FALSE)
dim(mHealth_1) # 161280, 24 columns
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
# This doesn't make any difference to the model accuracy - the accelerometer data is just as good a predictor of activity

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

train2 = data.frame(train1$Activity_Label_2, train1$Velocity_Chest, train1$Velocity_LAnkle, train1$Velocity_Arm)
dim(train2) # 17587, 4
train2 = data.frame(preProcess(train2))
dim(train2) #17587, 4

install.packages("tictoc")
library(tictoc)

tic("Running the ELM using three regular features, full dataset")
OSelm_train.formula(mHealth_1_labelled$Activity_Label_2~mHealth_1_labelled$Acc_Chest_X+mHealth_1_labelled$Acc_Chest_Y+mHealth_1_labelled$Acc_Chest_Z, mHealth_1_labelled,"classification", 40, "sig", 10, 10)
toc()
# Without pre-processing:
# 7.632s to train the model using 3 features and 40 hidden nodes, 79.90% accuracy
# 6.439s to train the model using 3 features and 10 hidden nodes, 80.49% accuracy

dim(train2) # 17587 columns, 4 rows
train2 = data.frame(preProcess(train2))

tic("Running the ELM using three velocity features, full dataset")
OSelm_train.formula(mHealth_1_labelled$Activity_Label_2~mHealth_1_labelled$Velocity_Chest+mHealth_1_labelled$Velocity_LAnkle+mHealth_1_labelled$Velocity_Arm, mHealth_1_labelled,"classification", 10, "sig", 10, 10)
toc()
# Without pre-processing:
# 73.7% accuracy, 6.595s to train the model using 3 features and 10 hidden nodes
# 73.7% accuracy, 7.871s to train the model using 3 features and 40 hidden nodes

tic("Running the ELM using three regular features using training data only")
OSelm_train.formula(train1$Activity_Label_2~train1$Acc_Chest_X+train1$Acc_Chest_Y+train1$Acc_Chest_Z, train1,"classification", 40, "sig", 10, 10)
toc()
# Without pre-processing:
# 46.3% accuracy, 1.694s to train the model using 3 features and 10 hidden nodes
# 82.2% accuracy, 2.009s to train the model using 3 features and 40 hidden nodes

tic("Running the ELM using three velocity features using training data only")
OSelm_train.formula(train1$Activity_Label_2~train1$Velocity_Chest+train1$Velocity_LAnkle+train1$Velocity_Arm, train1,"classification", 100, "sig", 10, 10)
toc()
# Without pre-processing:
# 46.3% accuracy, 1.694s to train the model using 3 features and 10 hidden nodes
# 34.8% accuracy, 2.624s to train the model using 3 features and 40 hidden nodes
# 35.3% accuracy, 8.407s to train the model using 3 features and 100 hidden nodes

train1 = data.frame(preProcess(train1))
tic("Running the ELM using three velocity features using training data only")
OSelm_train.formula(train1$Activity_Label_2~train1$Velocity_Chest+train1$Velocity_LAnkle+train1$Velocity_Arm, train1,"classification", 100, "sig", 10, 10)
toc()
# WITH pre-processing:
# 61.6% accuracy, 2.636s to train the model using 3 features and 10 hidden nodes
# 46.3% accuracy, 3.076s to train the model using 3 features and 40 hidden nodes
# 40.1% accuracy, 8.402s to train the model using 3 features and 100 hidden nodes

########################################################################################
install.packages("elmNN") # see https://cran.r-project.org/web/packages/elmNN/index.html
library(elmNN)

set.seed(1234)
smp_size  <- floor(0.50*nrow(mHealth_1_labelled))
train_index <- sample(seq_len(nrow(mHealth_1_labelled)),size = smp_size)

train <- mHealth_1_labelled[train_index,]
test <- mHealth_1_labelled[-train_index,]

tic("Training the model for 3 features from the training data")
model <- elmtrain(train$Activity_Label_2~train$Acc_Chest_X+train$Acc_Chest_Y+train$Acc_Chest_Z, data=train, nhid=50, actfun="sig")
toc()
# training time 1.159s

rawPrediction <- predict(model,newdata = NULL) # uses the training dataset, adds rawPrediction as a column
train <- cbind(train,rawPrediction)
print(model)

#Prediction values are a probability so we need to convert them to (0,1) in order to calculate the confusion matrix

normalized <- (rawPrediction-min(rawPrediction))/(max(rawPrediction)-min(rawPrediction))
train <- cbind(train,normalized)

#Calculate the train MSE
# Use a for loop to calculate the optimal value for the fitted(x) for which we choose (0 | 1)
# This takes each fitted probability and assigns a label of (0 | 1) depending on the value across the whole training vector
# The computes the training accuracy

trainMSE<-rep(0,20)
for(i in 1:20) {
  train$predictedLabel <- ifelse(train$normalized<=(0.05*i),0,1)
  trainMSE[i] <- mean(train$predictedLabel!=train$Activity_Label_2)
}
trainAccuracyRate=1-trainMSE
trainAccuracyRate

# 0.35 is the threshold which minimises the error based on fitted(x) -> (0 | 1)
# This gives an error rate of 91.6%, which seems reasonable for three features for this classifier 
# trainMSE_final is the error rate for a threshold vector value of 0.35 for 'inactive' vs 'active'

# Calculate the test error:

rawPrediction <- predict(model,newdata = test) # uses the training dataset, adds rawPrediction as a column
test <- cbind(test,rawPrediction)
View(test)
print(model)

normalized <- (rawPrediction-min(rawPrediction))/(max(rawPrediction)-min(rawPrediction))
test <- cbind(train,normalized)
View(test)

testMSE<-rep(0,20)
for(i in 1:20) {
  test$predictedLabel <- ifelse(test$normalized<=(0.05*i),0,1)
  testMSE[i] <- mean(test$predictedLabel!=test$Activity_Label_2)
}
testAccuracyRate=1-testMSE
testAccuracyRate

# Using the test data also gives an accuracy rate of 91.6% for a normalized threshold of 0.35

################################################################################################

# Try the model again using a different activation function with the same number of hidden nodes, say 50

tic("Training the model for 3 features from the training data")
model <- elmtrain(train$Activity_Label_2~train$Acc_Chest_X+train$Acc_Chest_Y+train$Acc_Chest_Z, data=train, nhid=50, actfun="radbas")
toc()
# training time 0.335s

rawPrediction <- predict(model,newdata = NULL) # uses the training dataset, adds rawPrediction as a column
train <- cbind(train,rawPrediction)
print(model)

#Prediction values are a probability so we need to convert them to (0,1) in order to calculate the confusion matrix

normalized <- (rawPrediction-min(rawPrediction))/(max(rawPrediction)-min(rawPrediction))
train <- cbind(train,normalized)

#Calculate the train MSE
# Use a for loop to calculate the optimal value for the fitted(x) for which we choose (0 | 1)
# This takes each fitted probability and assigns a label of (0 | 1) depending on the value across the whole training vector
# The computes the training accuracy

trainMSE<-rep(0,20)
for(i in 1:20) {
  train$predictedLabel <- ifelse(train$normalized<=(0.05*i),0,1)
  trainMSE[i] <- mean(train$predictedLabel!=train$Activity_Label_2)
}
trainAccuracyRate=1-trainMSE
trainAccuracyRate

# 0.40 is the threshold which minimises the error based on fitted(x) -> (0 | 1)
# This gives an error rate of 84.4%, which seems reasonable for three features for this classifier 
# trainMSE_final is the error rate for a threshold vector value of 0.40 for 'inactive' vs 'active'

# Calculate the test error:
rawPrediction <- predict(model,newdata = test) # uses the training dataset, adds rawPrediction as a column
test <- cbind(test,rawPrediction)
View(test)
print(model)

normalized <- (rawPrediction-min(rawPrediction))/(max(rawPrediction)-min(rawPrediction))
test <- cbind(train,normalized)
View(test)

testMSE<-rep(0,20)
for(i in 1:20) {
  test$predictedLabel <- ifelse(test$normalized<=(0.05*i),0,1)
  testMSE[i] <- mean(test$predictedLabel!=test$Activity_Label_2)
}
testAccuracyRate=1-testMSE
testAccuracyRate

# Using the test data also gives an accuracy rate of 84.4% for a normalized threshold of 0.40


# Now tune for the number of hidden neurons //FIXME this loop is not working ...
#########################################################################

set.seed(1234)
smp_size  <- floor(0.50*nrow(mHealth_1_labelled))
train_index <- sample(seq_len(nrow(mHealth_1_labelled)),size = smp_size)

train <- mHealth_1_labelled[train_index,]
test <- mHealth_1_labelled[-train_index,]
dim(train)

tic("Tuning for the number of hidden neurons using the training data and a sigmoid activation function")
trainAccuracyRate <- rep(0,8)
dim(trainAccuracyRate)
hidden_neurons = c(2,5,10,50,100,200,500,1000)
dim(hidden_neurons)
for (i in hidden_neurons) {
  model <- elmtrain(train$Activity_Label_2~train$Acc_Chest_X+train$Acc_Chest_Y+train$Acc_Chest_Z, data=train, nhid=hidden_neurons[i], actfun="sig")
  rawPrediction <- predict(model,newdata=NULL)
  normPrediction <- (rawPrediction-min(rawPrediction))/(max(rawPrediction)-min(rawPrediction))
  prediction <- ifelse(normPrediction<=0.45,0,1)
  trainAccuracyRate[i] <- mean(prediction==train$Activity_Label_2)
}
trainAccuracyRate
toc()
############################################################################


# Now trying a logistic binomial regression based on a number of factors, to understand which is the best predictor of activity level

#glm.fit=glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Weekly,family=binomial)
View(train)
glm.fit=glm(Activity_Label_2~Acc_Chest_X+Acc_Chest_Y+Acc_Chest_Z+ECG_1+ECG_2, data=train, family=binomial)
summary(glm.fit)
glm.probs.train=predict(glm.fit,data=train,type='response')
glm.probs.train[1:10]
glm.predict.train=rep(0,17587)
glm.predict.train[glm.probs.train>0.4]=1

train <- cbind(train,glm.predict.train)
View(train)

train_MSE <- mean(train$glm.predict.train!=train$Activity_Label_2)
train_MSE # 18.3% error





  



