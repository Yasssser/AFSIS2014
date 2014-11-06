#Seed number for cross validation
SEED=2014


#Loading Required Libraries
library(RWeka)#for PrincipalComponentAnalysis filter
library(wavelets)#for discret wavelet transforms
library(nnet)#training simple layer neural network
library(matrixStats)#calculate std of columns
library(pls)#multivariate regression
library(monmlp)#training multilayer perceptron
library(e1071)#training SVM
library(prospectr)#preprocessing 
library(kernlab)# training gaussian process
library(sqldf)#used for sampling dataset and cross validation split

RMSE=function(real,pred)
{
	nr=nrow(real)
	sm1=sqrt((1/nr)*(sum((real[,1]-pred[,1])^2)))
	return (sm1)
}

#Discrete Wavelet Transforms using Haar Algorithm
#DF1: input matrix for transform
#nTimes: number of iterations 
HaarTransform=function(DF1,nTimes=1)
{
	w =function(k)
	{
		s1=dwt(k, filter="haar")
		return (s1@V[[1]])
	}
	Smt=DF1
	for (i in 1:nTimes)
	{
		Smt=t(apply(Smt,1,w))
	}
	return (data.frame(Smt))
}

#Getting Derivatives 
#DF1: input matrix for transform
#D: Order 
Derivative=function(DF1,D=1)
{
	df1=t(diff(t(DF1), differences = D))	
	return(df1)
}

#train MLP on train and then predict test using weights gained by train
#train: train data frame for training
#test: test data frame to predict
#Other parameters are passed into monmlp function
#Result is predicted values for test 
GetMLPPreds=function(train,test,Labels,Iters=100,Hidden1=5,Hidden2=5,IWeights=0.5,N.ensemble=10,LTh = tansig, LTo = linear,LTh.prime = tansig.prime, LTo.prime = linear.prime,Seed=1)
{
	gc()
	set.seed(Seed)
	m1=monmlp.fit(as.matrix(train),as.matrix(Labels),scale.y=T,n.trials = 1, hidden1=Hidden1, hidden2=Hidden2, n.ensemble=N.ensemble,Th = LTh, To = LTo,
	 Th.prime = LTh.prime, To.prime = LTo.prime,iter.max=Iters,monotone=NULL, bag=F, init.weights = c(-(IWeights),IWeights),max.exceptions = 10,silent = T)
	pr1=monmlp.predict(as.matrix(test),weights=m1)
	rm(m1)
	gc()
	return (data.frame(pr1))	
}

#train simple layer neural network on train and then predict test using model gained by train
#train: train data frame for training
#test: test data frame to predict
#Other parameters are passed into nnet function
#Result is predicted values for test 
GetNNETPreds=function(train,test,Labels,Size=10,Rang=0.5,Decay=0.1,Iters=100,MaxWts=1500)
{
	set.seed(1)
	g1=nnet((Labels)~.,data=train,size=Size,linout=T,skip =T, rang = Rang, decay = Decay,MaxNWts = MaxWts, maxit = Iters,trace=F)
	pr1=predict(g1,test)
	rm(g1)
	gc()
	return (data.frame(pr1))	
}

#train SVM on train and then predict test using model gained by train
#train: train data frame for training
#test: test data frame to predict
#Other parameters are passed into SVM function in e1071 library
#Result is predicted values for test 
GetSVMPreds=function(train,test,Labels,Cost=10000)
{
	set.seed(1)
	s1=svm(data.frame(train),Labels,scale = F,cost = Cost)
	pr1=(predict(s1,data.frame(test)))
	return (data.frame(pr1))		
}

#train GaussianProcess on train and then predict test using model gained by train
#train: train data frame for training
#test: test data frame to predict
#Other parameters are passed into gausspr function in kernlab library
#Result is predicted values for test 
GetGaussPreds=function(train,test,Labels,Kernel='rbfdot',Kpar='automatic',Tol=0.05,Var=0.01)
{
	set.seed(1)
	v1=gausspr(data.frame(train), (Labels),type= NULL, kernel=Kernel,
	          kpar=Kpar, var=Var, variance.model = T, tol=Tol, cross=0, fit=F)
	pr1=(predict(v1,data.frame(test)))
	rm(v1)
	gc()
	return (data.frame(pr1))			
}

#train MVR on train and then predict test using model gained by train
#train: train data frame for training
#test: test data frame to predict
#Other parameters are passed into mvr function in pls library
#Result is predicted values for test 
GetMVRPreds=function(train,test,Labels,Ncomp=120,Scale=True)
{
	set.seed(1)
	v1=mvr(Labels~.,data=data.frame(train),ncomp=Ncomp, method = pls.options()$pcralg,scale = T)
	pr1=data.frame(predict(v1,data.frame(test)))
	pr1=data.frame(rowMeans(pr1))
	rm(v1)
	gc()
	return (data.frame(pr1))			
}


#Seprate data sets based on "Depth" variable and then train two seperate mlp models and then combine results
#train: train data frame for training
#test: test data frame to predict
#Other parameters are passed into monmlp function
#Result is predicted values for test 
GetMLPDepthPreds=function(train,test,Labels,Iters=100,Hidden1=5,Hidden2=5,IWeights=0.5,N.ensemble=10,LTh = tansig, LTo = linear,LTh.prime = tansig.prime, LTo.prime = linear.prime,Seed=1)
{
	trainD1=train[train[,'Depth']==1,]
	trainD2=train[train[,'Depth']==2,]
	testD1=test[test[,'Depth']==1,]
	testD2=test[test[,'Depth']==2,]
	prD1=GetMLPPreds(trainD1,testD1,MyTarget[train[,'Depth']==1],Iters=Iters,Hidden1=Hidden1,Hidden2=Hidden2,IWeights=IWeights,N.ensemble=N.ensemble,LTh = LTh, LTo = LTo,LTh.prime = LTh.prime, LTo.prime = LTo.prime,Seed=Seed)
	colnames(prD1)[1]='pr'
	prD2=GetMLPPreds(trainD2,testD2,MyTarget[train[,'Depth']==2],Iters=Iters,Hidden1=Hidden1,Hidden2=Hidden2,IWeights=IWeights,N.ensemble=N.ensemble,LTh = LTh, LTo = LTo,LTh.prime = LTh.prime, LTo.prime = LTo.prime,Seed=Seed)
	colnames(prD2)[1]='pr'
	prD=rbind(prD1,prD2)
	prD[test[,'Depth']==1,1]=prD1
	prD[test[,'Depth']==2,1]=prD2
	return (data.frame(prD))
}

#Seprate data sets based on "Depth" variable and then train two seperate svm models and then combine results
#train: train data frame for training
#test: test data frame to predict
#Other parameters are passed into svm function
#Result is predicted values for test 
GetSVMDepthPreds=function(train,test,Labels,Cost=10000)
{
	train=train
	test=test
	trainD1=train[train[,'Depth']==1,-ncol(train)]
	trainD2=train[train[,'Depth']==2,-ncol(train)]
	testD1=test[test[,'Depth']==1,-ncol(train)]
	testD2=test[test[,'Depth']==2,-ncol(train)]
	prD1=GetSVMPreds(trainD1,testD1,MyTarget[train[,'Depth']==1],Cost=Cost)
	colnames(prD1)[1]='pr'
	prD2=GetSVMPreds(trainD2,testD2,MyTarget[train[,'Depth']==2],Cost=Cost)
	colnames(prD2)[1]='pr'
	prD=rbind(prD1,prD2)
	prD[test[,'Depth']==1,1]=prD1
	prD[test[,'Depth']==2,1]=prD2
	return (data.frame(prD))
}

#Seprate data sets based on "Depth" variable and then train two seperate gaussian process models and then combine results
#train: train data frame for training
#test: test data frame to predict
#Other parameters are passed into gausspr function
#Result is predicted values for test 
GetGaussDepthPreds=function(train,test,Labels,Kernel='rbfdot',Kpar='automatic',Tol=0.05,Var=0.01)
{
	train=train
	test=test
	trainD1=train[train[,'Depth']==1,-ncol(train)]
	trainD2=train[train[,'Depth']==2,-ncol(train)]
	testD1=test[test[,'Depth']==1,-ncol(train)]
	testD2=test[test[,'Depth']==2,-ncol(train)]
	prD1=GetGaussPreds(trainD1,testD1,MyTarget[train[,'Depth']==1],Kernel=Kernel,Kpar=Kpar,Tol=Tol,Var=Var)
	colnames(prD1)[1]='pr'
	prD2=GetGaussPreds(trainD2,testD2,MyTarget[train[,'Depth']==2],Kernel=Kernel,Kpar=Kpar,Tol=Tol,Var=Var)
	colnames(prD2)[1]='pr'
	prD=rbind(prD1,prD2)
	prD[test[,'Depth']==1,1]=prD1
	prD[test[,'Depth']==2,1]=prD2
	return (data.frame(prD))
}

#Seprate data sets based on "Depth" variable and then train two seperate mvr process models and then combine results
#train: train data frame for training
#test: test data frame to predict
#Other parameters are passed into mvr function
#Result is predicted values for test 
GetMVRDepthPreds=function(train,test,Labels,Ncomp=120,Scale=True)
{
	train=train
	test=test
	trainD1=train[train[,'Depth']==1,-ncol(train)]
	trainD2=train[train[,'Depth']==2,-ncol(train)]
	testD1=test[test[,'Depth']==1,-ncol(train)]
	testD2=test[test[,'Depth']==2,-ncol(train)]
	prD1=GetMVRPreds(trainD1,testD1,MyTarget[train[,'Depth']==1],Ncomp=Ncomp,Scale=Scale)
	colnames(prD1)[1]='pr'
	prD2=GetMVRPreds(trainD2,testD2,MyTarget[train[,'Depth']==2],Ncomp=Ncomp,Scale=Scale)
	colnames(prD2)[1]='pr'
	prD=rbind(prD1,prD2)
	prD[test[,'Depth']==1,1]=prD1
	prD[test[,'Depth']==2,1]=prD2
	return (data.frame(prD))
}

#Calcule PCA using Weka PrincipalComponents filter
#df: input data frame 
#var: variance parameter for PCA
WekPCA=function(df,var)
{
	pc=make_Weka_filter('weka/filters/unsupervised/attribute/PrincipalComponents')
	d1=pc(df[,1]~.,data=df[,-1],control=c('-R',var))
	return (d1[,-ncol(d1)])
}

#Combine train and test and then get rank of features based on their standard deviation
#trainDF:train data frame
#testDF:train data frame
#result is rank of features
GetRanksBySD=function(trainDF,testDF)
{
	TAT=rbind(trainDF,testDF)
	rnk=rank(colSds(as.matrix(TAT)))
	return (rnk)
}

#Reading Dataset
TrainAndTestM=read.csv('training.csv')
TrainAndTestM[,3595]=as.numeric(TrainAndTestM[,3595])
#################
#################
#Sampling Dataset based on TMAP variable
TMAP=sqldf('select distinct TMAP from TrainAndTestM order by TMAP')
set.seed(SEED)
TMAPS=data.frame(TMAP[sample(nrow(TMAP)),])
TMAPS=cbind(1:nrow(TMAPS),TMAPS)
colnames(TMAPS)[1]='Ord'
colnames(TMAPS)[2]='TMAP'
ttt=sqldf('select i.*,j.Ord from TrainAndTestM i left join TMAPS j on (i.TMAP=j.TMAP)')
TrainAndTest=TrainAndTestM[order(ttt[,'Ord']),]
#calcule partial PCs of combined data set
#devide data set into 30 sub frames and then getting PCs 
#the combine sub-frames 
PC=list()
for (i in 1:30)
{
	j1=(i-1)*119+2
	j2=(i)*119+1
	if (i==30)
	{
		j2=3579
	}	
	temp1=TrainAndTest[,j1:j2]
	flush.console()
	PC[[i]]=WekPCA(cbind(TrainAndTest['Ca'],temp1),0.999)	
}
PComponents=PC[[1]]
for (i in 2:30)
{
	PComponents=cbind(PComponents,PC[[i]])
}

#Multiple Scatter Correction on spectral features(two phases) 
TrainAndTestReduced=msc(as.matrix(TrainAndTest[,2:3579]))
TrainAndTestReduced=msc(as.matrix(TrainAndTestReduced))

#First Derivatives
TrainAndTestReduced=Derivative(TrainAndTestReduced,1)

#Original data set(without transformation)
TrainAndTestOriginal=TrainAndTest[2:3600]

#Reduced Dataset(PCA,DWT)
TrainAndTestReduced=cbind(PComponents,data.frame(HaarTransform(TrainAndTestReduced,9)),TrainAndTest[,3580:3600])

#Creating data frame for submission
#submission=data.frame(PIDN=TestDataSet['PIDN'],Ca=1:nrow(TestDataSet),P=1:nrow(TestDataSet),pH=1:nrow(TestDataSet),SOC=1:nrow(TestDataSet),Sand=1:nrow(TestDataSet))
#k=2#Counter of submission columns
msm=0
nf=5
cat('Seed Number:',SEED,'\n')
for (TheTarget in c('Ca','P','pH','SOC','Sand'))
#for (TheTarget in c('pH'))
{
	cvsm=0
	cat (TheTarget,'.....................................................................\n')
	for (cv in 1:nf)
	{
		cat('CV',cv,':-------------------------------------\n')
		nd=floor(nrow(TrainAndTestReduced)/nf)
		testindexes=((cv-1)*nd+1):(cv*nd)
		#Retriving train and test data frames from original data set
		trainOriginal=TrainAndTestOriginal[-(testindexes),]
		testOriginal=TrainAndTestOriginal[(testindexes),]
		tstTRG=testOriginal[TheTarget]
		#Retriving train and test data frames from reduced data set	
		trainReduced=TrainAndTestReduced[-(testindexes),]
		testReduced=TrainAndTestReduced[(testindexes),]
			
		ThisTarget=trainOriginal[,TheTarget]
		MyTarget=trainOriginal[,TheTarget]
		
		#Saturation and log transform for "P" target
		if (TheTarget=='P')
		{
			MyTarget=ifelse(MyTarget>6,6,MyTarget)
			MyTarget=log(1+MyTarget)
		}
		
		trainReduced=trainReduced[,!colnames(trainReduced)%in% c('Ca','P','pH','SOC','Sand')]
		testReduced=testReduced[,!colnames(testReduced)%in% c('Ca','P','pH','SOC','Sand')]
		trainOriginal=trainOriginal[,!colnames(trainOriginal)%in% c('Ca','P','pH','SOC','Sand')]
		testOriginal=testOriginal[,!colnames(testOriginal)%in% c('Ca','P','pH','SOC','Sand')]
		
		#Training and Prediction phase for "Ca" target
		if (TheTarget=='Ca')
		{
			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			MyTrain=savitzkyGolay(MyTrain, p = 3, w = 21, m = 0)#savitzkyGolay filter from prospectr library
			MyTest=savitzkyGolay(MyTest, p = 3, w = 21, m = 0)
			MyTrain=cbind(MyTrain,trainOriginal[c(3583,3586,3587,3588,3589,3591,3594)])
			MyTest=cbind(MyTest,testOriginal[c(3583,3586,3587,3588,3589,3591,3594)])		
			Ca_SVM_Preds1=GetSVMPreds(MyTrain,MyTest,MyTarget,1000)
			cat('Ca_SVM_Preds1:',RMSE(data.frame(tstTRG),Ca_SVM_Preds1),'   ')
			flush.console();gc();cat('\n');
			
			flush.console()
			Ca_SVM_Preds2=GetSVMPreds(trainOriginal,testOriginal,MyTarget,Cost=10000)
			cat('Ca_SVM_Preds2:',RMSE(data.frame(tstTRG),Ca_SVM_Preds2),'   ')	
			flush.console();gc();cat('\n');

			flush.console()
			Ca_SVM_Preds3=GetSVMPreds(trainOriginal,testOriginal,MyTarget,5000)
			cat('Ca_SVM_Preds3:',RMSE(data.frame(tstTRG),Ca_SVM_Preds3),'   ')	
			flush.console();gc();cat('\n');

			flush.console()
			MyTrain=trainOriginal
			MyTest=testOriginal
			SSS=(rbind(MyTrain,MyTest))
			#get first 2000 features order by their standard deviation
			ordr=colnames(SSS)[order(colSds(as.matrix(SSS)),decreasing=T)]
			MyTrain=MyTrain[,ordr[1:2000]]		
			MyTest=MyTest[,ordr[1:2000]]			
			MyTrain=HaarTransform(Derivative(MyTrain),3)
			MyTest=HaarTransform(Derivative(MyTest),3)
			Ca_SVM_Preds4=GetSVMPreds(MyTrain,MyTest,MyTarget,10000)
			cat('Ca_SVM_Preds4:',RMSE(data.frame(tstTRG),Ca_SVM_Preds4),'   ')	
			flush.console();gc();cat('\n');

			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			#Get 2500 features with highest standard deviation
			rnk=GetRanksBySD(MyTrain,MyTest)
			MyTrain=MyTrain[,rnk<2500]		
			MyTest=MyTest[,rnk<2500]			
			MyTrain=cbind(HaarTransform(MyTrain,4),trainOriginal[c(3579:3581)])
			MyTest=cbind(HaarTransform(MyTest,4),testOriginal[c(3579:3581)])		
			#Get average of 10 different mlp model with different seed numbers
			Ca_MLP_Preds1=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=150,Hidden1=4,Hidden2=4,IWeights=0.5,Seed=1,N.ensemble=2)
			CNT=10
			for (sd in 2:CNT)
			{
				Ca_MLP_Preds1=Ca_MLP_Preds1+GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=150,Hidden1=4,Hidden2=4,IWeights=0.5,Seed=sd,N.ensemble=2)
				flush.console()
			}		
			Ca_MLP_Preds1=Ca_MLP_Preds1/CNT	
			cat('Ca_MLP_Preds1:',RMSE(data.frame(tstTRG),Ca_MLP_Preds1),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			MyTrain=savitzkyGolay(MyTrain, p = 3, w = 11, m = 0)
			MyTest=savitzkyGolay(MyTest, p = 3, w = 11, m = 0)
			MyTrain=cbind(HaarTransform(MyTrain,4),trainOriginal[c(3579:3581)])
			MyTest=cbind(HaarTransform((MyTest),4),testOriginal[c(3579:3581)])		
			Ca_MLP_Preds2=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=150,Hidden1=4,Hidden2=4,IWeights=0.5,Seed=1,N.ensemble=10)
			cat('Ca_MLP_Preds2:',RMSE(data.frame(tstTRG),Ca_MLP_Preds2),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=cbind(HaarTransform(trainOriginal[,1:3578],5),trainOriginal[,c(3579,3581)])
			MyTest=cbind(HaarTransform(testOriginal[,1:3578],5),testOriginal[,c(3579,3581)])
			Ca_MLP_Preds3=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=100,Hidden1=5,Hidden2=5,IWeights=0.5)		
			cat('Ca_MLP_Preds3:',RMSE(data.frame(tstTRG),Ca_MLP_Preds3),'   ')	
			flush.console();gc();cat('\n');

			flush.console()
			MyTrain=cbind(HaarTransform(Derivative(trainOriginal[,1:3578]),7),trainOriginal[,c(3579:3581)])
			MyTest=cbind(HaarTransform(Derivative(testOriginal[,1:3578]),7),testOriginal[,c(3579:3581)])
			Ca_MLP_Preds4=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=150,Hidden1=3,Hidden2=20,IWeights=0.6)		
			cat('Ca_MLP_Preds4:',RMSE(data.frame(tstTRG),Ca_MLP_Preds4),'   ')	
			flush.console();gc();cat('\n');

			flush.console()
			MyTrain=cbind(Derivative(HaarTransform(trainOriginal,4)),trainOriginal[,c(3579:3581,3594)])
			MyTest=cbind(Derivative(HaarTransform(testOriginal,4)),testOriginal[,c(3579:3581,3594)])
			Ca_MLP_Preds5=GetMLPDepthPreds(MyTrain,MyTest,MyTarget,Iters=100,Hidden1=5,Hidden2=5,IWeights=0.5)
			cat('Ca_MLP_Preds5:',RMSE(data.frame(tstTRG),Ca_MLP_Preds5),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=cbind(HaarTransform(trainOriginal,5),trainOriginal[,c(3579:3581,3594)])
			MyTest=cbind(HaarTransform(testOriginal,5),testOriginal[,c(3579:3581,3594)])
			Ca_MLP_Preds6=GetMLPDepthPreds(MyTrain,MyTest,MyTarget,Iters=100,Hidden1=5,Hidden2=5,IWeights=0.5)
			cat('Ca_MLP_Preds6:',RMSE(data.frame(tstTRG),Ca_MLP_Preds6),'   ')	
			flush.console();gc();cat('\n');

			flush.console()
			MyTrain=cbind(HaarTransform(trainOriginal[,1:3578],4),trainOriginal[,c(3579:3582)])
			MyTest=cbind(HaarTransform(testOriginal[,1:3578],4),testOriginal[,c(3579:3582)])
			Ca_MLP_Preds7=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=150,Hidden1=5,Hidden2=5,IWeights=0.5)		
			cat('Ca_MLP_Preds7:',RMSE(data.frame(tstTRG),Ca_MLP_Preds7),'   ')	
			flush.console();gc();cat('\n');
		
			flush.console()
			MyTrain=cbind(HaarTransform(trainOriginal[,1:3578],3),trainOriginal[,c(3579:3582)])
			MyTest=cbind(HaarTransform(testOriginal[,1:3578],3),testOriginal[,c(3579:3582)])
			Ca_MLP_Preds8=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=150,Hidden1=5,Hidden2=5,IWeights=0.5)		
			cat('Ca_MLP_Preds8:',RMSE(data.frame(tstTRG),Ca_MLP_Preds8),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			Ca_Gauss_Preds1=GetGaussPreds(trainOriginal,testOriginal,MyTarget,Kernel='rbfdot',Tol=0.05,Var=0.01)
			cat('Ca_Gauss_Preds1:',RMSE(data.frame(tstTRG),Ca_Gauss_Preds1),'   ')	
			flush.console();gc();cat('\n');

			flush.console()
			Ca_Gauss_Preds2=GetGaussPreds(trainReduced,testReduced,MyTarget,Kernel='rbfdot',Tol=0.05,Var=0.01)
			cat('Ca_Gauss_Preds2:',RMSE(data.frame(tstTRG),Ca_Gauss_Preds2),'   ')	
			flush.console();gc();cat('\n');

			flush.console()
			Ca_Gauss_Preds3=GetGaussPreds(trainOriginal,testOriginal,MyTarget,Kernel='polydot',Tol=0.001,Var=0.1)
			cat('Ca_Gauss_Preds3:',RMSE(data.frame(tstTRG),Ca_Gauss_Preds3),'   ')	
			flush.console();gc();cat('\n');
		
			flush.console()
			MyTrain=trainOriginal
			MyTest=testOriginal
			SSS=(rbind(MyTrain,MyTest))
			ordr=colnames(SSS)[order(colSds(as.matrix(SSS)),decreasing=T)]
			MyTrain=MyTrain[,ordr[1:2000]]		
			MyTest=MyTest[,ordr[1:2000]]			
			Ca_MVR_Preds1=GetMVRPreds(MyTrain,MyTest,MyTarget,120,True)
			cat('Ca_MVR_Preds1:',RMSE(data.frame(tstTRG),Ca_MVR_Preds1),'   ')	
			flush.console();gc();cat('\n');				

			flush.console()
			Ca_MVR_Preds2=GetMVRPreds(trainOriginal,testOriginal,MyTarget,100,True)
			cat('Ca_MVR_Preds2:',RMSE(data.frame(tstTRG),Ca_MVR_Preds2),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=HaarTransform(trainOriginal,5)
			MyTest=HaarTransform(testOriginal,5)
			Ca_NNET_Preds1=GetNNETPreds(MyTrain,MyTest,MyTarget,Size=10,Rang=0.5,Decay=0.1,Iters=100)		
			cat('Ca_NNET_Preds1:',RMSE(data.frame(tstTRG),Ca_NNET_Preds1),'   ')	
			flush.console();gc();cat('\n');
			
			#Combining predictions
			ThisPred=(100*Ca_SVM_Preds1+100*Ca_MLP_Preds1+100*Ca_MLP_Preds2+15*Ca_Gauss_Preds1+30*Ca_SVM_Preds2+100*Ca_SVM_Preds3+45*Ca_Gauss_Preds2+10*Ca_Gauss_Preds3+15*Ca_MVR_Preds1+10*Ca_SVM_Preds4+150*Ca_MLP_Preds3+50*Ca_MLP_Preds4+10*Ca_NNET_Preds1+5*Ca_MVR_Preds2+30*Ca_MLP_Preds5+30*Ca_MLP_Preds6+150*Ca_MLP_Preds7+50*Ca_MLP_Preds8)/1000		
			cat('Ca_Ensemble:',RMSE(data.frame(tstTRG),ThisPred),' \n  ')	
			
			flush.console();gc();cat('\n');
		}
		
		#Training and Prediction phase for "P" target
		if (TheTarget=='P')
		{
			flush.console()
			MyTrain=((trainOriginal[,1:3578]))
			MyTest=((testOriginal[,1:3578]))
			MyTrain=continuumRemoval(MyTrain, type='R',method='substraction')#continuumRemoval from prospectr library
			MyTest=continuumRemoval(MyTest, type='R',method='substraction')
			MyTrain=ifelse(is.na(MyTrain),1,MyTrain)
			MyTest=ifelse(is.na(MyTest),1,MyTest)
			MyTrain=cbind(MyTrain,trainOriginal[c(3579,3582)])
			MyTest=cbind(MyTest,testOriginal[c(3579,3582)])		
			P_SVM_Preds1=GetSVMPreds(MyTrain,MyTest,MyTarget,5000)
			P_SVM_Preds1[,1]=exp(P_SVM_Preds1[,1])-1
			cat('P_SVM_Preds1:',RMSE(data.frame(tstTRG),P_SVM_Preds1),'   ')	
			flush.console();gc();cat('\n');
		
			flush.console()
			MyTrain=cbind(trainOriginal[,1:3578],trainOriginal[,3579:3594])
			MyTest=cbind(testOriginal[,1:3578],testOriginal[,3579:3594])
			P_SVM_Preds2=GetSVMPreds(MyTrain,MyTest,MyTarget,5000)
			P_SVM_Preds2[,1]=exp(P_SVM_Preds2[,1])-1
			cat('P_SVM_Preds2:',RMSE(data.frame(tstTRG),P_SVM_Preds2),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=cbind(HaarTransform(trainOriginal[,1:3578],1),trainOriginal[c(3583,3594)])
			MyTest=cbind(HaarTransform(testOriginal[,1:3578],1),testOriginal[c(3583,3594)])
			P_SVM_Preds3=GetSVMDepthPreds(MyTrain,MyTest,MyTarget,Cost=1000)
			P_SVM_Preds3[,1]=exp(P_SVM_Preds3[,1])-1
			cat('P_SVM_Preds3:',RMSE(data.frame(tstTRG),P_SVM_Preds3),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=((trainOriginal[,1:3578]))
			MyTest=((testOriginal[,1:3578]))
			MyTrain=savitzkyGolay(MyTrain, p = 4, w = 11, m = 1)
			MyTest=savitzkyGolay(MyTest, p = 4, w = 11, m = 1)
			rnk=GetRanksBySD(MyTrain,MyTest)
			MyTrain=MyTrain[,rnk<3000]		
			MyTest=MyTest[,rnk<3000]			
			MyTrain=cbind(HaarTransform(MyTrain,4),trainOriginal[c(3579:3581,3594)])
			MyTest=cbind(HaarTransform(MyTest,4),testOriginal[c(3579:3581,3594)])		
			P_MLP_Preds1=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=150,Hidden1=5,Hidden2=0,IWeights=0.5)		
			P_MLP_Preds1[,1]=exp(P_MLP_Preds1[,1])-1
			cat('P_MLP_Preds1:',RMSE(data.frame(tstTRG),P_MLP_Preds1),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			MyTrain=savitzkyGolay(MyTrain, p = 4, w = 11, m = 1)
			MyTest=savitzkyGolay(MyTest, p = 4, w = 11, m = 1)
			rnk=GetRanksBySD(MyTrain,MyTest)
			MyTrain=MyTrain[,rnk<2500]		
			MyTest=MyTest[,rnk<2500]			
			MyTrain=cbind(HaarTransform(MyTrain,4),trainOriginal[c(3579:3581,3594)])
			MyTest=cbind(HaarTransform(MyTest,4),testOriginal[c(3579:3581,3594)])		
			P_MLP_Preds2=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=150,Hidden1=4,Hidden2=4,IWeights=0.6,Seed=1,N.ensemble=5)
			P_MLP_Preds2[,1]=exp(P_MLP_Preds2[,1])-1
			cat('P_MLP_Preds2:',RMSE(data.frame(tstTRG),P_MLP_Preds2),'   ')	
			flush.console();gc();cat('\n');	
			
			flush.console()
			MyTrain=cbind(Derivative(HaarTransform(trainOriginal,4)),trainOriginal[,c(3579:3581,3594)])
			MyTest=cbind(Derivative(HaarTransform(testOriginal,4)),testOriginal[,c(3579:3581,3594)])
			P_MLP_Preds3=GetMLPDepthPreds(MyTrain,MyTest,MyTarget,Iters=100,Hidden1=5,Hidden2=5,IWeights=0.5)
			P_MLP_Preds3[,1]=exp(P_MLP_Preds3[,1])-1
			cat('P_MLP_Preds3:',RMSE(data.frame(tstTRG),P_MLP_Preds3),'   ')	
			flush.console();gc();cat('\n');
		
			flush.console()
			MyTrain=HaarTransform(trainOriginal,5)
			MyTest=HaarTransform(testOriginal,5)
			P_MLP_Preds4=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=50,Hidden1=5,Hidden2=5,IWeights=0.6)		
			P_MLP_Preds4[,1]=exp(P_MLP_Preds4[,1])-1
			cat('P_MLP_Preds4:',RMSE(data.frame(tstTRG),P_MLP_Preds4),'   ')	
			flush.console();gc();cat('\n');

			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			MyTrain=cbind(HaarTransform(Derivative(MyTrain),2),trainOriginal[,c(3579,3593)])
			MyTest=cbind(HaarTransform(Derivative(MyTest),2),testOriginal[,c(3579,3593)])
			SSS=(rbind(MyTrain,MyTest))
			ordr=colnames(SSS)[order(colSds(as.matrix(SSS)),decreasing=T)]
			MyTrain=MyTrain[,ordr[1:450]]		
			MyTest=MyTest[,ordr[1:450]]			
			P_MLP_Preds5=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=50,Hidden1=5,Hidden2=5,IWeights=0.6)		
			P_MLP_Preds5[,1]=exp(P_MLP_Preds5[,1])-1
			cat('P_MLP_Preds5:',RMSE(data.frame(tstTRG),P_MLP_Preds5),'   ')	
			flush.console();gc();cat('\n');
		
			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			rnk=GetRanksBySD(MyTrain,MyTest)
			MyTrain=MyTrain[,rnk<2500]		
			MyTest=MyTest[,rnk<2500]			
			MyTrain=cbind(Derivative(HaarTransform(MyTrain,4)),trainOriginal[c(3579:3581)])
			MyTest=cbind(Derivative(HaarTransform(MyTest,4)),testOriginal[c(3579:3581)])		
			P_MLP_Preds6=GetMLPPreds(MyTrain[],MyTest,MyTarget,Iters=100,Hidden1=5,Hidden2=5,IWeights=0.5)		
			P_MLP_Preds6[,1]=exp(P_MLP_Preds6[,1])-1
			cat('P_MLP_Preds6:',RMSE(data.frame(tstTRG),P_MLP_Preds6),'   ')	
			flush.console();gc();cat('\n');

			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			rnk=GetRanksBySD(MyTrain,MyTest)
			MyTrain=MyTrain[,rnk<2500]		
			MyTest=MyTest[,rnk<2500]			
			MyTrain=cbind(Derivative(HaarTransform(MyTrain,4)),trainOriginal[c(3579:3581,3594)])
			MyTest=cbind(Derivative(HaarTransform(MyTest,4)),testOriginal[c(3579:3581,3594)])		
			P_MLP_Preds7=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=100,Hidden1=5,Hidden2=5,IWeights=0.5)		
			P_MLP_Preds7[,1]=exp(P_MLP_Preds7[,1])-1
			cat('P_MLP_Preds7:',RMSE(data.frame(tstTRG),P_MLP_Preds7),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			MyTrain=cbind(HaarTransform(MyTrain,4),trainOriginal[c(3579:3581)])
			MyTest=cbind(HaarTransform((MyTest),4),testOriginal[c(3579:3581)])		
			P_MVR_Preds1=GetMVRPreds(MyTrain,MyTest,MyTarget,Ncomp=200,Scale=True)
			P_MVR_Preds1[,1]=exp(P_MVR_Preds1[,1])-1
			cat('P_MVR_Preds1:',RMSE(data.frame(tstTRG),P_MVR_Preds1),'   ')	
			flush.console();gc();cat('\n');
				
			ThisPred=(70*P_SVM_Preds1+70*P_MVR_Preds1+70*P_MLP_Preds1+100*P_MLP_Preds3+70*P_SVM_Preds2+70*P_SVM_Preds3+50*P_MLP_Preds4+50*P_MLP_Preds5+50*P_MLP_Preds6+200*P_MLP_Preds7)/800
			cat('P_Ensemble:',RMSE(data.frame(tstTRG),ThisPred),'\n   ')	
				
		}
		
		#Training and Prediction phase for "pH" target
		if(TheTarget=='pH')
		{	
			flush.console()
			MyTrain=cbind(HaarTransform(trainOriginal[,1:3578],1),trainOriginal[c(3583,3594)])
			MyTest=cbind(HaarTransform(testOriginal[,1:3578],1),testOriginal[c(3583,3594)])
			pH_SVM_Preds1=GetSVMDepthPreds(MyTrain,MyTest,MyTarget,Cost=1000)
			cat('pH_SVM_Preds1:',RMSE(data.frame(tstTRG),pH_SVM_Preds1),'   ')	
			flush.console();gc();cat('\n');
		
			flush.console()
			MyTrain=cbind(trainOriginal[,1:3578],trainOriginal[,3579:3594])
			MyTest=cbind(testOriginal[,1:3578],testOriginal[,3579:3594])
			pH_SVM_Preds2=GetSVMPreds(MyTrain,MyTest,MyTarget,5000)
			cat('pH_SVM_Preds2:',RMSE(data.frame(tstTRG),pH_SVM_Preds2),'   ')	
			flush.console();gc();cat('\n');
		
			flush.console()
			MyTrain=cbind(HaarTransform(trainOriginal[,1:3578],5),trainOriginal[,c(3579,3581)])
			MyTest=cbind(HaarTransform(testOriginal[,1:3578],5),testOriginal[,c(3579,3581)])
			pH_MLP_Preds1=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=100,Hidden1=5,Hidden2=5,IWeights=0.5)		
			cat('pH_MLP_Preds1:',RMSE(data.frame(tstTRG),pH_MLP_Preds1),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=cbind(Derivative(HaarTransform(trainOriginal,4)),trainOriginal[,c(3579:3581,3594)])
			MyTest=cbind(Derivative(HaarTransform(testOriginal,4)),testOriginal[,c(3579:3581,3594)])
			pH_MLP_Preds2=GetMLPDepthPreds(MyTrain,MyTest,MyTarget,Iters=100,Hidden1=5,Hidden2=5,IWeights=0.5)
			cat('pH_MLP_Preds2:',RMSE(data.frame(tstTRG),pH_MLP_Preds2),'   ')	
			flush.console();gc();cat('\n');
		
			flush.console()
			MyTrain=cbind(HaarTransform(trainOriginal[,1:3578],4),trainOriginal[,c(3579:3582)])
			MyTest=cbind(HaarTransform(testOriginal[,1:3578],4),testOriginal[,c(3579:3582)])
			pH_MLP_Preds3=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=150,Hidden1=5,Hidden2=5,IWeights=0.5)		
			cat('pH_MLP_Preds3:',RMSE(data.frame(tstTRG),pH_MLP_Preds3),'   ')	
			flush.console();gc();cat('\n');
		
			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			rnk=GetRanksBySD(MyTrain,MyTest)
			MyTrain=MyTrain[,rnk<2500]		
			MyTest=MyTest[,rnk<2500]			
			MyTrain=cbind(Derivative(HaarTransform(MyTrain,4)),trainOriginal[c(3579:3581)])
			MyTest=cbind(Derivative(HaarTransform(MyTest,4)),testOriginal[c(3579:3581)])		
			pH_MLP_Preds4=GetMLPPreds(MyTrain[],MyTest,MyTarget,Iters=100,Hidden1=5,Hidden2=5,IWeights=0.5)		
			cat('pH_MLP_Preds4:',RMSE(data.frame(tstTRG),pH_MLP_Preds4),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			rnk=GetRanksBySD(MyTrain,MyTest)
			MyTrain=MyTrain[,rnk<2500]		
			MyTest=MyTest[,rnk<2500]			
			MyTrain=cbind(Derivative(HaarTransform(MyTrain,4)),trainOriginal[c(3579:3581,3594)])
			MyTest=cbind(Derivative(HaarTransform(MyTest,4)),testOriginal[c(3579:3581,3594)])		
			pH_MLP_Preds5=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=100,Hidden1=5,Hidden2=5,IWeights=0.5)		
			cat('pH_MLP_Preds5:',RMSE(data.frame(tstTRG),pH_MLP_Preds5),'   ')	
			flush.console();gc();cat('\n');
			
			ThisPred=(70*pH_MLP_Preds1+70*pH_MLP_Preds2+50*pH_SVM_Preds1+50*pH_MLP_Preds3+50*pH_SVM_Preds2+70*pH_MLP_Preds4+70*pH_MLP_Preds5)/430
			cat('pH_Ensemble:',RMSE(data.frame(tstTRG),ThisPred),'\n   ')	
			
		}
		
		#Training and Prediction phase for "SOC" target
		if (TheTarget=='SOC')
		{
			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			MyTrain=cbind(MyTrain,trainOriginal[c(3581,3590,3591)])
			MyTest=cbind(MyTest,testOriginal[c(3581,3590,3591)])		
			SOC_SVM_Preds1=GetSVMPreds(MyTrain,MyTest,MyTarget,10000)
			cat('SOC_SVM_Preds1:',RMSE(data.frame(tstTRG),SOC_SVM_Preds1),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=cbind(trainOriginal[,1:3578],trainOriginal[,3579:3594])
			MyTest=cbind(testOriginal[,1:3578],testOriginal[,3579:3594])
			SOC_SVM_Preds2=GetSVMPreds(MyTrain,MyTest,MyTarget,5000)
			cat('SOC_SVM_Preds2:',RMSE(data.frame(tstTRG),SOC_SVM_Preds2),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			MyTrain=savitzkyGolay(MyTrain, p = 3, w = 11, m = 0)
			MyTest=savitzkyGolay(MyTest, p = 3, w = 11, m = 0)
			rnk=GetRanksBySD(MyTrain,MyTest)
			MyTrain=MyTrain[,rnk<2500]		
			MyTest=MyTest[,rnk<2500]			
			MyTrain=cbind(HaarTransform(MyTrain,3),trainOriginal[c(3579:3581,3594)])
			MyTest=cbind(HaarTransform(MyTest,3),testOriginal[c(3579:3581,3594)])		
			SOC_MLP_Preds1=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=150,Hidden1=3,Hidden2=3,IWeights=0.5,Seed=1,N.ensemble=20)
			cat('SOC_MLP_Preds1:',RMSE(data.frame(tstTRG),SOC_MLP_Preds1),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			MyTrain=savitzkyGolay(MyTrain, p = 3, w = 11, m = 0)
			MyTest=savitzkyGolay(MyTest, p = 3, w = 11, m = 0)
			MyTrain=cbind(HaarTransform(MyTrain,3),trainOriginal[c(3579:3581,3594)])
			MyTest=cbind(HaarTransform(MyTest,3),testOriginal[c(3579:3581,3594)])		
			SOC_MLP_Preds2=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=150,Hidden1=3,Hidden2=3,IWeights=0.5,Seed=1,N.ensemble=10)	
			cat('SOC_MLP_Preds2:',RMSE(data.frame(tstTRG),SOC_MLP_Preds2),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console();
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			MyTrain=savitzkyGolay(MyTrain, p = 3, w = 11, m = 0)
			MyTest=savitzkyGolay(MyTest, p = 3, w = 11, m = 0)
			rnk=GetRanksBySD(MyTrain,MyTest)
			MyTrain=MyTrain[,rnk<2500]		
			MyTest=MyTest[,rnk<2500]			
			MyTrain=cbind(HaarTransform(MyTrain,4),trainOriginal[c(3579:3581)])
			MyTest=cbind(HaarTransform(MyTest,4),testOriginal[c(3579:3581)])		
			SOC_MLP_Preds3=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=150,Hidden1=4,Hidden2=4,IWeights=0.5,Seed=1,N.ensemble=10)
			cat('SOC_MLP_Preds3:',RMSE(data.frame(tstTRG),SOC_MLP_Preds3),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=cbind(HaarTransform(Derivative(trainOriginal[,1:3578]),6),trainOriginal[,c(3579:3581)])
			MyTest=cbind(HaarTransform(Derivative(testOriginal[,1:3578]),6),testOriginal[,c(3579:3581)])
			SOC_MLP_Preds4=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=100,Hidden1=4,Hidden2=0,IWeights=0.5)		
			cat('SOC_MLP_Preds4:',RMSE(data.frame(tstTRG),SOC_MLP_Preds4),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=HaarTransform(trainOriginal,5)
			MyTest=HaarTransform(testOriginal,5)
			SOC_MLP_Preds5=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=50,Hidden1=5,Hidden2=5,IWeights=0.6)		
			cat('SOC_MLP_Preds5:',RMSE(data.frame(tstTRG),SOC_MLP_Preds5),'   ')	
			flush.console();gc();cat('\n');

			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			rnk=GetRanksBySD(MyTrain,MyTest)
			MyTrain=MyTrain[,rnk<2500]		
			MyTest=MyTest[,rnk<2500]			
			MyTrain=cbind(Derivative(HaarTransform(MyTrain,4)),trainOriginal[c(3579:3581)])
			MyTest=cbind(Derivative(HaarTransform(MyTest,4)),testOriginal[c(3579:3581)])		
			SOC_MLP_Preds6=GetMLPPreds(MyTrain[],MyTest,MyTarget,Iters=100,Hidden1=5,Hidden2=5,IWeights=0.5)		
			cat('SOC_MLP_Preds6:',RMSE(data.frame(tstTRG),SOC_MLP_Preds6),'   ')	
			flush.console();gc();cat('\n');

			ThisPred=(100*SOC_SVM_Preds1+50*SOC_MLP_Preds3+50*SOC_MLP_Preds2+50*SOC_MLP_Preds1+70*SOC_SVM_Preds2+100*SOC_MLP_Preds4+60*SOC_MLP_Preds5+20*SOC_MLP_Preds6)/500
			cat('SOC_Ensemble:',RMSE(data.frame(tstTRG),ThisPred),'\n   ')	
		}
		
		#Training and Prediction phase for "Sand" target
		if (TheTarget=='Sand')
		{
			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			MyTrain=savitzkyGolay(MyTrain, p = 3, w = 11, m = 0)
			MyTest=savitzkyGolay(MyTest, p = 3, w = 11, m = 0)
			MyTrain=cbind(MyTrain,trainOriginal[c(3581,3583,3585,3586,3588,3590,3591:3592,3594)])
			MyTest=cbind(MyTest,testOriginal[c(3581,3583,3585,3586,3588,3590,3591:3592,3594)])		
			Sand_SVM_Preds1=GetSVMPreds(MyTrain,MyTest,MyTarget,5000)
			cat('Sand_SVM_Preds1:',RMSE(data.frame(tstTRG),Sand_SVM_Preds1),'   ')	
			flush.console();gc();cat('\n');
				
			flush.console()
			Sand_SVM_Preds2=GetSVMPreds(trainOriginal,testOriginal,MyTarget,Cost=10000)#OK
			cat('Sand_SVM_Preds2:',RMSE(data.frame(tstTRG),Sand_SVM_Preds2),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=trainOriginal
			MyTest=testOriginal
			SSS=(rbind(MyTrain,MyTest))
			MyTrain=SSS[(1:nrow(trainOriginal)),]
			MyTest=SSS[-(1:nrow(trainOriginal)),]
			ordr=colnames(SSS)[order(colSds(as.matrix(SSS)),decreasing=T)]
			MyTrain=MyTrain[,ordr[1:2000]]		
			MyTest=MyTest[,ordr[1:2000]]		
			MyTrain=HaarTransform(Derivative(MyTrain),3)
			MyTest=HaarTransform(Derivative(MyTest),3)
			Sand_SVM_Preds3=GetSVMPreds(MyTrain,MyTest,MyTarget,10000)
			cat('Sand_SVM_Preds3:',RMSE(data.frame(tstTRG),Sand_SVM_Preds3),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			SSS=(rbind(MyTrain,MyTest))
			MyTrain=SSS[(1:nrow(trainOriginal)),]
			MyTest=SSS[-(1:nrow(trainOriginal)),]
			ordr=colnames(SSS)[order(colSds(as.matrix(SSS)),decreasing=T)]
			MyTrain=MyTrain[,ordr[1:1500]]		
			MyTest=MyTest[,ordr[1:1500]]		
			MyTrain=HaarTransform(Derivative(MyTrain),3)
			MyTest=HaarTransform(Derivative(MyTest),3)
			Sand_SVM_Preds4=GetSVMPreds(MyTrain,MyTest,MyTarget,10000)
			cat('Sand_SVM_Preds4:',RMSE(data.frame(tstTRG),Sand_SVM_Preds4),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=HaarTransform(trainOriginal[,1:3578],4)
			MyTest=HaarTransform(testOriginal[,1:3578],4)
			MyTrain=savitzkyGolay(MyTrain, p = 2, w = 3, m = 1)
			MyTest=savitzkyGolay(MyTest, p = 2, w = 3, m = 1)
			MyTrain=cbind(MyTrain,trainOriginal[c(3579,3593)])
			MyTest=cbind(MyTest,testOriginal[c(3579,3593)])		
			Sand_MLP_Preds1=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=150,Hidden1=4,Hidden2=4,IWeights=0.6,Seed=1,N.ensemble=10)
			cat('Sand_MLP_Preds1:',RMSE(data.frame(tstTRG),Sand_MLP_Preds1),'   ')	
			flush.console();gc();cat('\n');
		
			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			MyTrain=HaarTransform(MyTrain,4)
			MyTest=HaarTransform(MyTest,4)
			MTT=rbind(MyTrain,MyTest)
			MTT=WekPCA(MTT,0.9995)
			MyTrain=MTT[1:nrow(MyTrain),]
			MyTest=MTT[-(1:nrow(MyTrain)),]
			MyTrain=cbind(MyTrain,trainOriginal[c(3579:3581,3594)])
			MyTest=cbind(MyTest,testOriginal[c(3579:3581,3594)])	
			Sand_MLP_Preds2=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=200,Hidden1=5,Hidden2=5,IWeights=0.5,Seed=1,N.ensemble=30)
			cat('Sand_MLP_Preds2:',RMSE(data.frame(tstTRG),Sand_MLP_Preds2),'   ')	
			flush.console();gc();cat('\n');
					
			flush.console()
			MyTrain=HaarTransform(trainOriginal,5)
			MyTest=HaarTransform(testOriginal,5)
			Sand_MLP_Preds3=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=50,Hidden1=4,Hidden2=0,IWeights=0.5)		
			cat('Sand_MLP_Preds3:',RMSE(data.frame(tstTRG),Sand_MLP_Preds3),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=HaarTransform(trainOriginal,5)
			MyTest=HaarTransform(testOriginal,5)
			Sand_MLP_Preds4=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=50,Hidden1=5,Hidden2=5,IWeights=0.6)		
			cat('Sand_MLP_Preds4:',RMSE(data.frame(tstTRG),Sand_MLP_Preds4),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=trainOriginal[,1:3578]
			MyTest=testOriginal[,1:3578]
			MyTrain=cbind(HaarTransform(Derivative(MyTrain),2),trainOriginal[,c(3579,3593)])
			MyTest=cbind(HaarTransform(Derivative(MyTest),2),testOriginal[,c(3579,3593)])
			SSS=(rbind(MyTrain,MyTest))
			MyTrain=SSS[(1:nrow(trainOriginal)),]
			MyTest=SSS[-(1:nrow(trainOriginal)),]
			ordr=colnames(SSS)[order(colSds(as.matrix(SSS)),decreasing=T)]
			MyTrain=MyTrain[,ordr[1:450]]		
			MyTest=MyTest[,ordr[1:450]]		
			Sand_MLP_Preds5=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=50,Hidden1=5,Hidden2=5,IWeights=0.6)		
			cat('Sand_MLP_Preds5:',RMSE(data.frame(tstTRG),Sand_MLP_Preds5),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=cbind(Derivative(HaarTransform(trainOriginal,4)),trainOriginal[,c(3579:3581,3594)])
			MyTest=cbind(Derivative(HaarTransform(testOriginal,4)),testOriginal[,c(3579:3581,3594)])
			Sand_MLP_Preds6=GetMLPDepthPreds(MyTrain,MyTest,MyTarget,Iters=100,Hidden1=5,Hidden2=5,IWeights=0.5)
			cat('Sand_MLP_Preds6:',RMSE(data.frame(tstTRG),Sand_MLP_Preds6),'   ')	
			flush.console();gc();cat('\n');
		
			flush.console()
			MyTrain=cbind(HaarTransform(trainOriginal[,1:3578],4),trainOriginal[,c(3579:3582)])
			MyTest=cbind(HaarTransform(testOriginal[,1:3578],4),testOriginal[,c(3579:3582)])
			Sand_MLP_Preds7=GetMLPPreds(MyTrain,MyTest,MyTarget,Iters=150,Hidden1=5,Hidden2=5,IWeights=0.5)		
			cat('Sand_MLP_Preds7:',RMSE(data.frame(tstTRG),Sand_MLP_Preds7),'   ')	
			flush.console();gc();cat('\n');
		
			flush.console()
			Sand_Gauss_Preds1=GetGaussPreds(trainOriginal,testOriginal,MyTarget,Kernel='rbfdot',Tol=0.05,Var=0.01)
			cat('Sand_Gauss_Preds1:',RMSE(data.frame(tstTRG),Sand_Gauss_Preds1),'   ')	
			flush.console();gc();cat('\n');

			flush.console()
			Sand_Gauss_Preds2=GetGaussPreds(trainReduced,testReduced,MyTarget,Kernel='rbfdot',Tol=0.05,Var=0.01)
			cat('Sand_Gauss_Preds2:',RMSE(data.frame(tstTRG),Sand_Gauss_Preds2),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			Sand_Gauss_Preds3=GetGaussPreds(trainOriginal,testOriginal,MyTarget,Kernel='polydot',Tol=0.001,Var=0.1)
			cat('Sand_Gauss_Preds3:',RMSE(data.frame(tstTRG),Sand_Gauss_Preds3),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=trainOriginal
			MyTest=testOriginal
			SSS=(rbind(MyTrain,MyTest))
			ordr=colnames(SSS)[order(colSds(as.matrix(SSS)),decreasing=T)]
			MyTrain=MyTrain[,ordr[1:2000]]		
			MyTest=MyTest[,ordr[1:2000]]			
			Sand_MVR_Preds1=GetMVRPreds(MyTrain,MyTest,MyTarget,120,True)
			cat('Sand_MVR_Preds1:',RMSE(data.frame(tstTRG),Sand_MVR_Preds1),'   ')	
			flush.console();gc();cat('\n');
			
			flush.console()
			MyTrain=HaarTransform(trainOriginal,5)
			MyTest=HaarTransform(testOriginal,5)
			Sand_NNET_Preds1=GetNNETPreds(MyTrain,MyTest,MyTarget,Size=10,Rang=0.5,Decay=0.1,Iters=100)		
			cat('Sand_NNET_Preds1:',RMSE(data.frame(tstTRG),Sand_NNET_Preds1),'   ')	
			flush.console();gc();cat('\n');
			
			ThisPred=(15*Sand_Gauss_Preds1+30*Sand_SVM_Preds2+10*Sand_Gauss_Preds2+20*Sand_Gauss_Preds3+15*Sand_MVR_Preds1+50*Sand_SVM_Preds3+50*Sand_SVM_Preds4+10*Sand_MLP_Preds3+10*Sand_NNET_Preds1+70*Sand_MLP_Preds4+120*Sand_MLP_Preds5+30*Sand_MLP_Preds6+60*Sand_MLP_Preds7+100*Sand_SVM_Preds1+100*Sand_MLP_Preds2+100*Sand_MLP_Preds1)/790
			cat('Sand_Ensemble:',RMSE(data.frame(tstTRG),ThisPred),'   ')	
		}

		ThisPred[,1]=ifelse(ThisPred[,1]<min(ThisTarget),min(ThisTarget),ThisPred[,1])
		ThisPred[,1]=ifelse(ThisPred[,1]>max(ThisTarget),max(ThisTarget),ThisPred[,1])
		flush.console()
		cvsm=cvsm+RMSE(data.frame(tstTRG),ThisPred)
		cat('\n')
	}
	cat('  CV:',cvsm/nf,'\n')
	msm=msm+cvsm/nf
	flush.console();gc();
}

cat('OverallCV:',msm/5)
