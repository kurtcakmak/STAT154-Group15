library(e1071)
library(caret)
library(mlbench)
library(ROCR)

data = read.csv("~/desktop/train_final.csv",header = T)
# filter out words that occur fewer than 50 times in total
colsum = colSums(data)
filtered_df = data[,colsum>50]


#training and test sets
x = filtered_df[,-2362]
y = as.factor(filtered_df$biaoqian)


# validation set approach to choose the best cost
sample_df = filtered_df[sample(3505),]
rownames(sample_df) = c(1:3505)
test = sample_df[1:700,]
test.x = test[,-2362]
test.y = test$biaoqian
train = sample_df[701:3505,]
train.y = train$biaoqian

# use caret to choose the between linear, radial, and polynomial
# prepare training scheme
control <- trainControl(method="cv", number=5)
#CV MSE using svm
modelSvm1 <- train(x=x, y=y, method="svmLinear", trControl=control, metric = 'Accuracy')
modelSvm1
# accuracy: 0.7169722
# C = 1
modelSvm2 = train(x=x, y=y, method="svmRadial", trControl=control, metric = 'Accuracy')
modelSvm2
#C     Accuracy   Kappa    
#0.25  0.4990039  0.2631591
#0.50  0.5629204  0.3647671
#1.00  0.6393933  0.4854136
#sigma = 0.001293333 and C = 1

modelSvm3 = train(x=x, y=y, method="svmPoly", trControl=control, metric = 'Accuracy')
modelSvm3


svm001 = svm(biaoqian~., data = train, cost=0.01, type='C-classification',kernel = 'linear')
# 0.76 accuracy
#Confusion Matrices
svm_pred = predict(svm001, filtered_df[,-2362])
table(filtered_df$biaoqian, svm_pred)
confus2 = confusion.matrix(svm_pred, filtered_df$biaoqian)
confus2$overall[1]
#Compare 3 models:
resamps <- resamples(list(Linear = modelSvm1, Poly = modelSvm3, Radial = modelSvm2))
summary(resamps)
bwplot(resamps, metric = "Accuracy")
densityplot(resamps, metric = "Accuracy")


# 5-fold CV
folds <- createFolds(filtered_df, k = 5)
split_up <- lapply(folds, function(ind, dat) dat[ind,], dat = filtered_df)

# can use this to do a boxplot for each value of c
# rate: 10, 
rate = list()
svm_cv = function(c) {
  misclass = c()
  for (i in 1:5) {
    svm=svm(biaoqian~., data = filtered_df[-as.integer(rownames(split_up[[i]])),], cost=c, type='C-classification',kernel = 'linear')
    misclass[i] = sum(predict(svm, split_up[[i]][,-2362])!=split_up[[i]][,2362])/nrow(split_up[[i]])
  }
  return (mean(misclass))
}
boxplot(rate[[1]])

fit= svm_cv(0.00001)

fit7= svm_cv(0.0001)

fit1= svm_cv(0.001)
#0.2920379
fit2=svm_cv(0.01)
#0.2291622
fit3= svm_cv(0.1)
# 0.2628
fit4 = svm_cv(1)
# 0.2789786
fit5= svm_cv(10)
# 0.2789786
fit6 = svm_cv(100)
# 0.2789786

# find top 100, 200, 300, 400, 500 important features with training set and use them to refit model 
# to find accuracy on test set

control <- trainControl(method="cv", number=5)
#CV MSE using svm
model <- train(biaoqian~., data=train, method="svmLinear", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
print(importance)
# plot importance
plot(importance)
important500 = rownames(importance[[1]])[1:100]

# test
imp_train = train[,important500]
imp_test = test[,important500]

modelSvm1 <- train(y = as.factor(train.y), x = imp_train, method="svmLinear", trControl=control, metric = 'Accuracy')
modelSvm1
predictions = predict(modelSvm1, imp_test)
table(predictions,test.y)
confus2 = confusionMatrix(predictions, test.y)
confus2$overall[1]

svm0001 = svm(biaoqian~., data = train, cost=0.001, type='C-classification',kernel = 'linear')
# 0.6657143
svm001 = svm(biaoqian~., data = train, cost=0.01, type='C-classification',kernel = 'linear')
# 0.76 accuracy
svm01 = svm(biaoqian~., data = train, cost=0.1, type='C-classification',kernel = 'linear')
# 0.7171429 
svm1 = svm(biaoqian~., data = train, cost=1, type='C-classification',kernel = 'linear')
# 0.6957143 
svm10 = svm(biaoqian~., data = train, cost=10, type='C-classification',kernel = 'linear')
# 0.6957143
svm100 = svm(biaoqian~., data = train, cost=100, type='C-classification',kernel = 'linear')
# 0.6957143

svm_pred = predict(svm0001, test.x)
table(svm_pred,test.y)
confus2 = confusionMatrix(svm_pred, test.y)
confus2$overall[1]
