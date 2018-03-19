#importing the TRAIN data
train = read.csv("E:/EXTRA/@AV/DATAHACK- ROUND2/train_jDb5RBj.csv")

#importing the TEST data
test = read.csv("E:/EXTRA/@AV/DATAHACK- ROUND2/test_dan2xFI.csv")

#checking if any NA values are there
colSums(is.na(train))  #NO NA VALUES 
colSums(is.na(test))  #NO NA VALUES

#checking the percentage of TARGET variable
prop.table(table(train$Purchase))*100

training = train[,-1]
testing = test[,-1]

training$Purchase = as.factor(training$Purchase)
levels(training$Purchase) = c("NO", "YES")

#Creating data partition
library(caret)
index = createDataPartition(training$Purchase, times = 1, list = F, p = 0.7)
train_new = training[index,]
test_new = training[-index,]


library(dplyr) #for data manipulation
library(caret) #for model building
library(DMwR) #for smote implementation
library(purrr) #for functional programming(map)
library(pROC) #for AUC calculations

#Set up control function for training
ctrl = trainControl(method = "repeatedcv",
                    number = 10,
                    repeats = 5,
                    summaryFunction = twoClassSummary,
                    classProbs = T)
#ORIGINAL
set.seed(2969)
orig_fit = caret::train(Purchase ~ ., 
                        data = train_new,
                        method = "gbm",
                        metric = "ROC",
                        trControl = ctrl)

#AUC
pred1 = predict(orig_fit, test_new, type = "prob")[,"YES"]
auc(test_new$Purchase, pred1)

#Create modelweights ( they sum to one)
model_weights = ifelse(train_new$Purchase == "NO",
                       (1/table(train_new$Purchase)[1])*0.5,
                       (1/table(train_new$Purchase)[2])*0.5)

#Use the smae seed to ensure same cross-validation splits
ctrl$seeds = orig_fit$control$seeds

#Buid weighted model

weighted_fit = caret::train(Purchase ~ ., 
                            data = train_new,
                            method = "gbm",
                            verbose = F,
                            weights = model_weights,
                            metric = "ROC",
                            trControl = ctrl)

#Build DOWN SAMPLED model
ctrl$sampling = "down"

down_fit = caret::train(Purchase ~ ., 
                        data = train_new,
                        method = "gbm",
                        verbose = F,
                        metric = "ROC",
                        trControl = ctrl)

#Build UP SAMPLED model
ctrl$sampling = "up"

up_fit = caret::train(Purchase ~ ., 
                      data = train_new,
                      method = "gbm",
                      verbose = F,
                      metric = "ROC",
                      trControl = ctrl)

#Build SMOTE model
ctrl$sampling = "smote"

smote_fit = caret::train(Purchase ~ ., 
                         data = train_new,
                         method = "gbm",
                         verbose = F,
                         metric = "ROC",
                         trControl = ctrl)

#AUCs
pred1 = predict(orig_fit, test_new, type = "prob")[,"YES"]
auc(test_new$Purchase, pred1)

pred2 = predict(weighted_fit, test_new, type = "prob")[,"YES"]
auc(test_new$Purchase, pred2)

pred3 = predict(down_fit, test_new, type = "prob")[,"YES"]
auc(test_new$Purchase, pred3)

pred4 = predict(up_fit, test_new, type = "prob")[,"YES"]
auc(test_new$Purchase, pred4)

pred5 = predict(smote_fit, test_new, type = "prob")[,"YES"]
auc(test_new$Purchase, pred5)

#with grid tuning
gbmGrid =  expand.grid(interaction.depth = c(1,5,9),
                       n.trees = (1:30)*50,
                       shrinkage = 0.1,
                       n.minobsinnode = 20)

#Build DOWN SAMPLED model
ctrl$sampling = "down"

down_fit_2 = caret::train(Purchase ~ ., 
                        data = train_new,
                        method = "gbm",
                        verbose = F,
                        metric = "ROC",
                        trControl = ctrl,
                        tuneGrid = gbmGrid)

#Build UP SAMPLED model
ctrl$sampling = "up"

up_fit_2 = caret::train(Purchase ~ ., 
                      data = train_new,
                      method = "gbm",
                      verbose = F,
                      metric = "ROC",
                      trControl = ctrl,
                      tuneGrid = gbmGrid)

#Build SMOTE model
ctrl$sampling = "smote"

smote_fit_2 = caret::train(Purchase ~ ., 
                         data = train_new,
                         method = "gbm",
                         verbose = F,
                         metric = "ROC",
                         trControl = ctrl,
                         tuneGrid = gbmGrid)

#AUCs

pred3_2 = predict(down_fit_2, test_new, type = "prob")[,"YES"]
auc(test_new$Purchase, pred3_2)

pred4_2 = predict(up_fit_2, test_new, type = "prob")[,"YES"]
auc(test_new$Purchase, pred4_2)

pred5_2 = predict(smote_fit_2, test_new, type = "prob")[,"YES"]
auc(test_new$Purchase, pred5_2)

#Build UP SAMPLED model
ctrl$sampling = "up"

up_fit_2 = caret::train(Purchase ~ ., 
                        data = training,
                        method = "gbm",
                        verbose = F,
                        metric = "ROC",
                        trControl = ctrl,
                        tuneGrid = gbmGrid)


#prediction
pred = predict(up_fit, testing, type = "prob")[,"YES"]

#
submission = data.frame(test$ID, pred)
colnames(submission) = c("ID", "Purchase")

write.csv(x = submission, 
          file = "E:/EXTRA/@AV/DATAHACK- ROUND2/submission/sub0801_7.csv", row.names = F)

