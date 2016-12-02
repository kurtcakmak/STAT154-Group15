rm(list=ls())
data <- read.csv("~/desktop/train_final.csv",sep=',',header = T)
test <- read.csv("~/desktop/test.csv",sep=',',header = T)
#colnames(data)[1:28] = as.character((1:28))
matrix = as.matrix(data)
len_of_email=rowSums(data[,1:(p-1)])
unique_words=rowSums(sign(data[,1:(p-1)]))
data$len_of_email=len_of_email
data$unique_words=unique_words
test$biaoqian <- NULL

# try tf-idf
train = data[,-which(colsum<20)]
attach(train)
train$biaoqian = as.factor(train$biaoqian)
colnames(train)[112]='dot'
rf.data = randomForest(biaoqian~.-biaoqian, data=train, mtry=100,ntree=300)
idf = log(3506/colSums(sign(train[,1:4806]))+1)
tfidf = train
for (i in 1:4806){
  tfidf[,i]=train[,i]*idf[i]
}
attach(tfidf)
rf.dfidf = randomForest(biaoqian~.-biaoqian, data=tfidf, mtry=100,ntree=300)



length(which(colsum<(i+1)))-length(which(colsum<i))
matrix = cbind(sign(matrix[,1:(p-1)]),matrix[,p])
p=as.numeric( ncol(data))

colsum1 = array(0,(p-1))
colsum2 = array(0,(p-1))
colsum3 = array(0,(p-1))
colsum4 = array(0,(p-1))
colsum5 = array(0,(p-1))
for (i in 1:3505){
  if (matrix[i,p]==1){colsum1 = colsum1 + matrix[i,-p]}
  if (matrix[i,p]==2){colsum2 = colsum2 + matrix[i,-p]}
  if (matrix[i,p]==3){colsum3 = colsum3 + matrix[i,-p]}
  if (matrix[i,p]==4){colsum4 = colsum4 + matrix[i,-p]}
  if (matrix[i,p]==5){colsum5 = colsum5 + matrix[i,-p]}
}
colsum = colSums(matrix[,-p])
COLsum = as.data.frame(rbind(colsum1/685,colsum2/1023,colsum3/1241,colsum4/275,colsum5/281,colsum))

table = as.data.frame(rbind(colsum1/685,colsum2/1023,colsum3/1241,colsum4/275,colsum5/281,colsum))



library(Matrix)
#try somthing
class1 = data[which(data$biaoqian==1),]
class2 = data[which(data$biaoqian==2),]
class3 = data[which(data$biaoqian==3),]
class4 = data[which(data$biaoqian==4),]
class5 = data[which(data$biaoqian==5),]

#3500 0.03  f = 0.02  bool4 0.02  bool5 0.7
#3000 0.05  f = 0.03  bool4 0.03  bool5 1.5  (36)
#2500 0.06  f = 0.04  bool4 0.04  bool5 2.3  (48)
#2000 0.07  f = 0.05  bool4 0.05  bool5 3
#1500 0.10  f = 0.07  bool4 0.07  bool5 6
#1000 0.15  f = 0.1   bool4 0.09  bool5 10
vector = c()
for (i in 1:(p-1)){
  f = 0.07
  bool2 = (sum(COLsum[1:5,i]<0.1)==5)  # this is strong
  bool3 = (nnzero(class1[,i])<f*685)&&(nnzero(class2[,i])<f*1023)&&(nnzero(class3[,i])<f*1241)&&(nnzero(class4[,i])<f*275)&&(nnzero(class5[,i])<f*281)
  bool4 = (max(table[1:5,i]) - min(table[1:5,i]) < 0.07)
  bool5 = (var(table[1:5,i])*10000<6)
  # bool2 = F
  # bool3 = F
  # bool4 = F
  # bool5 = F
  if ((colsum[i]<20)||bool2||bool3||bool4||bool5){
    vector=c(vector,i)
  }
}
frequency = COLsum[,-vector]
sthbigger=data[,-vector]
sth2500=data[,-vector]
sth3000=data[,-vector]
sth1500=data[,-vector]
sth3500=data[,-vector]
sth3500$biaoqian=as.factor(data$biaoqian)



attach(sth1500)
rf.sth1500 = randomForest(biaoqian~.-biaoqian, data=sth1500, mtry=34,ntree=300)
#22.28%
attach(sthbigger)
rf.sthbigger = randomForest(biaoqian~.-biaoqian, data=sthbigger, mtry=40,ntree=300)
#21.91%
attach(sth2500)
rf.sth2500 = randomForest(biaoqian~.-biaoqian, data=sth2500, mtry=44,ntree=300)
#22.17%
attach(sth3000)
rf.sth3000 = randomForest(biaoqian~.-biaoqian, data=sth1500, mtry=49,ntree=300)
#21.77%
attach(sth3500)
rf.sth3500 = randomForest(biaoqian~.-biaoqian, data=sth3500, mtry=110,ntree=300)
#22.45%
rf.sth2000=rf.sthbigger
plot(c(0,300),c(0.2,0.45),type = 'n',xlab='# of trees',ylab='OOB error rate',main='OOB error for different feature size')
lines(1:300,rf.sth1500$err.rate[,1],'s',col = 'blue')
lines(1:300,rf.sth2000$err.rate[,1],'s',col = 'red')
lines(1:300,rf.sth2500$err.rate[,1],'s',col = 'green')
lines(1:300,rf.sth3000$err.rate[,1],'s',col = 'yellow')
lines(1:300,rf.sth3500$err.rate[,1],'s',col = 'brown')

# lines(1:300,rf.sth3500$err.rate[,1],'s',col = 'black')
legend(220,0.45,c("size 1161","size 1638","size 1983","size 2387","size 3280") , 
       lty=1, col=c('blue', 'red', 'green',' yellow','brown'), bty='n', cex=1)

a = cbind(rf.sth1500$err.rate[,1],rf.sth2000$err.rate[,1],rf.sth2500$err.rate[,1],rf.sth3000$err.rate[,1])
sth2000=sthbigger
sthbigger = sth3000
rf.sthbigger = rf.sth3000
len_of_email=rowSums(data[,1:(p-1)])
unique_words=rowSums(sign(data[,1:(p-1)]))
sthbigger$len_of_email=len_of_email
sthbigger$unique_words=unique_words
sthbigger$biaoqian = as.factor(sthbigger$biaoqian)
attach(sthbigger)
rf.data1 = randomForest(biaoqian~.-biaoqian, data=sthbigger, mtry=11,ntree=300)

rf.data2 = randomForest(biaoqian~.-biaoqian, data=sthbigger, mtry=49,ntree=300)
rf.data3 = randomForest(biaoqian~.-biaoqian, data=sthbigger, mtry=98,ntree=300)
rf.data4 = randomForest(biaoqian~.-biaoqian, data=sthbigger, mtry=1192,ntree=300)
plot(c(0,300),c(0.2,0.5),type = 'n',xlab='# of trees',ylab='OOB error rate',main='OOB error for different m')
lines(1:300,rf.data1$err.rate[,1],'s',col = 'blue')
lines(1:300,rf.data2$err.rate[,1],'s',col = 'red')
lines(1:300,rf.data3$err.rate[,1],'s',col = 'green')
lines(1:300,rf.data4$err.rate[,1],'s',col = 'brown')

# lines(1:300,rf.sth3500$err.rate[,1],'s',col = 'black')
legend(220,0.5,c("m = log2(p)","m = sqrt(p)","m = 2sqrt(p)","m = p/2") , 
       lty=1, col=c('blue', 'red', 'green','brown'), bty='n', cex=1)

rf.data5 = randomForest(biaoqian~.-biaoqian, data=sthbigger, mtry=49,ntree=150)
rf.data6 = randomForest(biaoqian~.-biaoqian, data=sthbigger, mtry=98,ntree=150)

rf.final = randomForest(biaoqian~.-biaoqian, data=sthbigger, mtry=1192,ntree=150)
rf.final
importance = rf.data4$importance
x = sort(importance, decreasing = T)
index300 = which(importance>=x[300])
index500 = which(importance>=x[500])
index800 = which(importance>=x[800])

#first remove label and then add them back because the label is not last column
sthbigger$biaoqian=NULL
sthbigger$biaoqian = as.factor(data$biaoqian)

important300 = sthbigger[,rownames(importance)[index300]]
important500 = sthbigger[,rownames(importance)[index500]]
important800 = sthbigger[,rownames(importance)[index800]]
important500$biaoqian = as.factor(data$biaoqian)
important300$biaoqian = as.factor(data$biaoqian)
important800$biaoqian = as.factor(data$biaoqian)
attach(important300)
attach(important500)
write.table(important500,file="important500.csv")
attach(important800)
rf = randomForest(biaoqian~.-biaoqian, data=important300, mtry=150,ntree=150)
rf
rf = randomForest(biaoqian~.-biaoqian, data=important500, mtry=250,ntree=150)
rf
rf = randomForest(biaoqian~.-biaoqian, data=important800, mtry=400,ntree=150)
rf

library(randomForest)
set.seed(15)
attach(sthbigger)
shuffle = sthbigger[sample(3505),]
#****************Cross Validation*********************#
CVData = shuffle[,rownames(importance)[index500]]
CVData$biaoqian=as.factor(data$biaoqian)
accuracy =array(0,5)
for (k in 1:5){
  test = CVData[((k-1)*701+1):(k*701),]
  train = CVData[-(((k-1)*701+1):(k*701)),]
  cvrf.data = randomForest(biaoqian~.-biaoqian,data=train, mtry=35,ntree=300)
  Y.test = predict(cvrf.data,newdata = test)
  accuracy[k] = length(which(Y.test == test[,ncol(test)]))/701
}
CV_accuracy = mean(accuracy)
#****************Cross Validation*********************#



#**********************Tuning mtry************************#
# install.packages('caret')
library(caret)
sthbigger$biaoqian = as.factor(data$biaoqian)
attach(sthbigger)
control <- trainControl(method="repeatedcv", number=5, repeats=1)
set.seed(15)
metric <- "Accuracy"
mtry <- c(40,70,100,200,409,818)
tunegrid <- expand.grid(.mtry=mtry)
ptm <- proc.time()
rf_random2 <- train(biaoqian~., data=sthbigger, method="rf",ntree = 300, nodesize=10,metric=metric, tuneGrid=tunegrid, trControl=control)
proc.time() - ptm
print(rf_random)
plot(rf_random)
#**********************Tuning mtry************************#


#*******************************K-means***********************************#
km.out=kmeans(important100[,1:100], 5, nstart = 20)
#******************************************************************************#


