#importing the TRAIN data
train = read.csv("/DATAHACK- ROUND2/train_jDb5RBj.csv")

#importing the TEST data
test = read.csv("/DATAHACK- ROUND2/test_dan2xFI.csv")

#checking if any NA values are there
colSums(is.na(train))  #NO NA VALUES 
colSums(is.na(test))  #NO NA VALUES

#checking the percentage of TARGET variable
prop.table(table(train$Purchase))*100

training = train[,-1]
testing = test[,-1]

training = as.data.frame(lapply(training, as.numeric))
testing = as.data.frame(lapply(testing, as.numeric))

training$Purchase = as.factor(training$Purchase)
levels(training$Purchase) = c("NO", "YES")

training_xgb = xgb.DMatrix(data = data.matrix(training), label = training$Purchase - 1)
testing_xgb = xgb.DMatrix(data = data.matrix(testing), label = testing$Purchase - 1)

#xgboost
library(xgboost)
xgb = xgboost(data = training_xgb, label = training$Purchase, eta = 0.1, max_depth = 15,
              nround = 25, subsample = 0.5, colsample_bytree = 0.5, seed = 2696,
              eval_metric = "error", obejctive = "binary:logistic", nthread = 3)

#prediction
pred = predict(xgb, testing) #, type = "prob")[,"YES"]

#
submission = data.frame(test$ID, pred)
colnames(submission) = c("ID", "Purchase")

write.csv(x = submission, 
          file = "E:/EXTRA/@AV/DATAHACK- ROUND2/submission/sub0801_T.csv", row.names = F)

