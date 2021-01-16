#Project 2
get.folds = function(n, k) {
  n.fold = ceiling(n / k)
  fold.ids.raw = rep(1:k, times = n.fold)
  fold.ids = fold.ids.raw[1:n]
  folds.rand = fold.ids[sample.int(n)]
  return(folds.rand)
}

getMSPE = function(y, y.hat) {
  resid = y - y.hat
  resid.sq = resid^2
  SSPE = sum(resid.sq)
  MSPE = SSPE / length(y)
  return(MSPE)
}

rescale = function(x1, x2) {
  for(col in 1:ncol(x1)) {
    a = min(x2[,col])
    b = max(x2[,col])
    x1[,col] = (x1[,col] - a) / (b - a)
  }
  x1
}

scale.1 <- function(x1,x2) {
  for(col in 1:ncol(x1)) {
    a <- mean(x2[,col])
    b <- sd(x2[,col])
    x1[,col] <- (x1[,col]-a)/b
  }
  x1
}


data = read.csv("P2Data2020.csv")

#Random Forest Tuning
library(randomForest)
set.seed(46685326, kind = "Mersenne-Twister")
perm <- sample(x = nrow(data))
set1 <- data[which(perm <= 3 * nrow(data)/4), ]
set2 <- data[which(perm > 3 * nrow(data)/4), ]

y.train = set1[,1]

all.mtrys = c(2, 4, 6, 10, 18)
all.nodesize = c(1, 3, 5, 7, 10)
all.pars.rf = expand.grid(mtry = all.mtrys, nodesize = all.nodesize)
n.pars = nrow(all.pars.rf)

M = 5

all.OOB.rf = array(0, dim = c(M, n.pars))
names.pars = apply(all.pars.rf, 1, paste0, collapse = "-")
colnames(all.OOB.rf) = names.pars

for(i in 1:n.pars) {
  print(paste0(i, " of ", n.pars))
  
  this.mtry = all.pars.rf[i, "mtry"]
  this.nodesize = all.pars.rf[i, "nodesize"]
  
  for(j in 1:M) {
    this.fit.rf = randomForest(Y ~ ., data = set1, mtry = this.mtry, nodesize = this.nodesize)
    
    this.pred.rf = predict(this.fit.rf)
    this.err.rf = mean(y.train != this.pred.rf)
    
    all.OOB.rf[j, i] = this.err.rf
  }
}

boxplot(all.OOB.rf, las = 2)
rel.OOB.rf = apply(all.OOB.rf, 1, function(w) w/min(w))
boxplot(t(rel.OOB.rf), las = 2, main = "RF Relative Error")


#NNET Tuning
library(nnet)
set.seed(46685326, kind = "Mersenne-Twister")
perm <- sample(x = nrow(data))
set1 <- data[which(perm <= 3 * nrow(data)/4), ]
set2 <- data[which(perm > 3 * nrow(data)/4), ]

x.train.rescale = rescale(set1[,-1], set1[,-1])
x.valid.rescale = rescale(set2[,-1], set1[,-1])
y.train = set1[,1]
y.valid = set2[,1]
y.train.num = class.ind(y.train)

all.sizes = c(1, 3, 5, 7, 9)
all.decays = c(0, 0.001, 0.01, 0.1, 1)
all.pars = expand.grid(size = all.sizes, decay = all.decays)
n.pars = nrow(all.pars)
par.names = apply(all.pars, 1, paste, collapse = "-")

k = 10
m = 5

CV.misclass = array(0, dim = c(k, n.pars))
colnames(CV.misclass) = par.names
folds = get.folds(n, k)
for(i in 1:k) {
  print(paste0(i, " of ", k))
  
  data.train = data[folds != i,]
  x.train = data.train[, -1]
  x.train.rescale = rescale(x.train, x.train)
  y.train = data.train[, 1]
  y.train.num = class.ind(y.train)
  
  data.valid = data[folds == i,]
  x.valid = data.valid[, -1]
  x.valid.rescale = rescale(x.valid, x.train)
  y.valid = data.valid[, 1]
  y.valid.num = class.ind(y.valid)
  
  for(j in 1:n.pars) {
    print("Doing each pairs")
    this.size = all.pars[j, "size"]
    this.decay = all.pars[i, "decay"]
    MSE.best = Inf
    
    for(l in 1:m) {
      print("Refitting models")
      this.nnet = nnet(x.train.rescale, y.train.num, size = this.size, decay = this.decay, maxit = 2000, softmax = TRUE, trace = FALSE)
      this.MSE = this.nnet$value
      if(this.MSE < MSE.best) {
        nnet.best = this.nnet
      }
    }
    
    pred.nnet.best = predict(nnet.best, x.valid.rescale, type = "class")
    this.mis.CV = mean(y.valid != pred.nnet.best)
    CV.misclass[i, j] = this.mis.CV
  }
}
boxplot(CV.misclass, las = 2)
rel.CV.misclass = apply(CV.misclass, 1, function(w) w/min(w))
boxplot(t(rel.CV.misclass), las = 2)

#Model Comparison
library(MASS)
library(klaR)
library(randomForest)
set.seed(5348725)

n = nrow(data)
k = 10
folds = get.folds(n, k)

all.models = c("LDA", "QDA", "NB00", "NB01", "NB10", "NB11", "RF", "NNET")
all.mis = array(0, dim = c(k, length(all.models)))
colnames(all.mis) = all.models

for(i in 1:k) {
  print(paste0(i, " of ", k))
  
  data.train = data[folds != i,]
  x.train = data.train[, -1]
  x.train.DA = scale.1(data.train[, -1], data.train[,- 1])
  x.train.rescale = rescale(x.train, x.train)
  y.train = data.train[, 1]
  y.train.num = class.ind(y.train)
  
  data.valid = data[folds == i,]
  x.valid = data.valid[, -1]
  x.valid.DA = scale.1(data.valid[, -1], data.train[, -1])
  x.valid.rescale = rescale(x.valid, x.train)
  y.valid = data.valid[, 1]
  y.valid.num = class.ind(y.valid)
  
  fit.lda = lda(x.train.DA, y.train)
  pred.lda = predict(fit.lda, x.valid.DA)$class
  mis.lda = mean(y.valid != pred.lda)
  all.mis[i, "LDA"] = mis.lda
  
  fit.qda = qda(x.train.DA, y.train)
  pred.qda = predict(fit.qda, x.valid.DA)$class
  mis.qda = mean(y.valid != pred.qda)
  all.mis[i, "QDA"] = mis.qda
  
  fit.NB00 = NaiveBayes(x.train, y.train, usekernel = FALSE)
  pred.NB00 = predict(fit.NB00, x.valid)$class
  mis.NB00 = mean(y.valid != pred.NB00)
  all.mis[i, "NB00"] = mis.NB00

  fit.NB01 = NaiveBayes(x.train, y.train, usekernel = TRUE)
  pred.NB01 = predict(fit.NB01, x.valid)$class
  mis.NB01 = mean(y.valid != pred.NB01)
  all.mis[i, "NB01"] = mis.NB01

  fit.PCA = prcomp(x.train, scale = TRUE)
  x.train.PC = fit.PCA$x
  x.valid.PC = predict(fit.PCA, data.valid)

  fit.NB10 = NaiveBayes(x.train.PC, y.train, usekernel = FALSE)
  pred.NB10 = predict(fit.NB10, x.valid.PC)
  pred.NB10 = pred.NB10$class
  mis.NB10 = mean(y.valid != pred.NB10)
  all.mis[i, "NB10"] = mis.NB10

  fit.NB11 = NaiveBayes(x.train.PC, y.train, usekernel = TRUE)
  pred.NB11 = predict(fit.NB11, x.valid.PC)
  pred.NB11 = pred.NB11$class
  mis.NB11 = mean(y.valid != pred.NB11)
  all.mis[i, "NB11"] = mis.NB11
  
  fit.rf = randomForest(Y ~ ., data = data.train, mtry = 2, nodesize = 1)
  pred.rf = predict(fit.rf, data.valid)
  mis.rf = mean(y.valid != pred.rf)
  all.mis[i, "RF"] = mis.rf
  
  MSE.best = Inf
  m = 10
  for(j in 1:m) {
   this.nnet = nnet(x.train.rescale, y.train.num, size = 3, decay = 1, maxit = 10000, softmax = TRUE, trace = FALSE)
   this.MSE = this.nnet$value
   if(this.MSE < MSE.best) {
     nnet.best = this.nnet
   }
   pred.nnet = predict(nnet.best, x.valid.rescale, type = "class")
   mis.nnet = mean(y.valid != pred.nnet)
   all.mis[i, "NNET"] = mis.nnet
  }
}

boxplot(all.mis)

all.rmis = apply(all.mis, 1, function(w) {
  best = min(w)
  return(w / best)
})
all.rmis = t(all.rmis)

boxplot(all.rmis)

#Final Prediction Machine
library(randomForest)
data = read.csv("P2Data2020.csv")
testData2020 = read.csv("P2Data2020testX.csv")
set.seed(7828297)
fit.rf = randomForest(Y ~ ., data = data, mtry = 2, nodesize = 1)
pred.rf = predict(fit.rf, testData2020)
write.table(pred.rf, "Project2Prediction.txt", sep = ",", row.names = F, col.names =
              F)

#Test Error and Confusion Matrix
library(randomForest)
set.seed(46685326, kind = "Mersenne-Twister")
perm <- sample(x = nrow(data))
set1 <- data[which(perm <= 3 * nrow(data)/4), ]
set2 <- data[which(perm > 3 * nrow(data)/4), ]
y.valid = set2[,1]
fit.rf = randomForest(Y ~ ., data = set1, mtry = 2, nodesize = 1)
pred.rf = predict(fit.rf, set2)
mis.rf = mean(y.valid != pred.rf)
table(y.valid, pred.rf, dnn = c("Observed", "Predicted"))
