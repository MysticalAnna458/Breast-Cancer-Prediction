install.packages("readr")
library(readr)
install.packages("corrplot")
library(corrplot)
install.packages("lattice")
library(lattice)
install.packages("ggplot2")
library(ggplot2)
install.packages("ggfortify")
library(ggfortify)
install.packages("neuralnet")
library(neuralnet)
install.packages("NeuralNetTools")
library(NeuralNetTools)
install.packages("caret")
library(caret)
install.packages("nnet")
library(nnet)
install.packages("NeuralNetTools")
library(NeuralNetTools)
install.packages("descr")
library(descr)
install.packages("cluster")
library(cluster)
install.packages("corrplot")
library(corrplot)
install.packages("InformationValue")
library(InformationValue)
library(devtools)
install_github('sinhrks/ggfortify')
library(ggfortify)
library(ggplot2)
install.packages("MASS")
library(MASS)
bcp = read.csv(file.choose(), header = TRUE)
head(bcp)
summary(bcp)
ncol(bcp)
bcp = bcp[,c(2:32)]
bcp
ncol(bcp)
bcp_cor = cor(bcp[,-c(1)])
bcp_cor
corrplot(bcp_cor, order = "hclust",tl.cex=1, addrect=7)
ncol(bcp_cor)
bcp_cor2 = bcp[,-findCorrelation(bcp_cor, cutoff=0.9)]
str(bcp_cor2)
ncol(bcp_cor2)
distance = dist(bcp_cor)
print(distance)
hc.c = hclust(distance)
plot(hc.c)
group = cutree(hc.c,7)
plot(silhouette(cutree(hc.c,7),distance))
# Scree plot
wss <- (nrow(bcp_cor)-1)*sum(apply(bcp_cor,2,var))
for (i in 2:29) wss[i] <- sum(kmeans(bcp_cor, centers=i)$withinss)
plot(1:29, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")
#post principle component analysis
pospro_bcp = prcomp(bcp_cor2,scale=TRUE,center=TRUE)
summary(pospro_bcp)
pospro_bcp_var= pospro_bcp$sdev^2
print(pospro_bcp_var, digits =3)
pos_pv = pospro_bcp_var/sum(pospro_bcp_var)
cumul_pos_pv = cumsum(pos_pv)
cumul_pos_pv
pospv_table =tibble(comp= seq(1:ncol(bcp_cor2)),pos_pv,cumul_pos_pv)

# this graph gives us idea about how what PC's are are most influential in the data
ggplot(pospv_table, aes(x=comp,y=cumul_pos_pv ))+geom_point(col="blue")+geom_abline(intercept = 0.95, slope = 0)

autoplot(pospro_bcp, data = bcp,  colour = 'diagnosis',loadings = FALSE, loadings.label = TRUE, loadings.colour = "blue")

# prediction
bcp_pred = predict(pospro_bcp,newdata = bcp_cor2)
str(bcp_pred)

bcpndata = cbind(diagnosis = bcp$diagnosis,bcp_pred)
str(bcpndata)
bcpndataf = data.frame((bcpndata))
str(bcpndataf)
bcpndataf$diagnosis = as.factor(bcpndataf$diagnosis)
# Logistic Regression
logistic = glm(diagnosis~.,data =bcpndataf,family = "binomial")
summary(logistic)

pred1 = predict(logistic, testing, type="response")
pred1
optcutoff = optimalCutoff(testing$diagnosis,pred1)
p1 = ifelse(pred1>optcutoff,1,0)
tb = table(p1,testing$diagnosis)
tb
accuracy2 = sum(diag(tb))/sum(tb)
accuracy2

stepAIC(logistic,direction="backward")
# Neural Networks
set.seed(678)
index = createDataPartition(bcpndataf$diagnosis, p =0.75, list =FALSE)
index
training = bcpndataf[index,]
testing = bcpndataf[-index,]
str(testing)
testing
nn = nnet(diagnosis~., data=training, size =7)
plotnet(nn, cex_val =.8,max_sp=T,circle_cex=5,circle_col = 'red')

pred_nn = predict(nn,testing, type="class")
str(pred_nn)

Table = CrossTable(testing$diagnosis,pred_nn,prop.chisq=F,prop.r=F,prop.c=F,dnn=c("Actual Diagnosis","Predict Diagnosis"))
Table
