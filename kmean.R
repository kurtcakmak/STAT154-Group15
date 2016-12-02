# Kmeans Clustering
library(randomForest)

df=read.csv("~/desktop/kmean_data.csv",header=TRUE)
label = as.factor(df$biaoqian)
x = df[,-1360]

rf = randomForest(y = label, x = x, mtry=80,ntree=300)
importance = rf$importance
# top 100 feature
index = which(importance>4.51)
index1 = which(importance>15)
important100 = df[,rownames(importance)[index]]
important10 = df[,rownames(importance)[index1]]
colnames(important10)
km.out=kmeans(important100[,1:100], 5)
km.out$tot.withinss
km.out$withinss
# try several times and this is the minimum

km.out$cluster

#335899
#8037.525  77723.587  30516.000 136944.646  82677.270

a = cbind(km.out$cluster,label)
table(label,km.out$cluster)
sum(km.out$cluster==5)
