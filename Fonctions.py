"""This is the document py including the functions to realise the binary classification in machine learning and is developped by Xiayue Shen, Xiran Zhang, Zuoyu Zhang"""

#Import some function libraries and related data processing support libraries
from scipy.io import arff
import pandas as pd
from pandas.core.reshape.melt import to_numeric

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import operator

#Import the libraries scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import NuSVC

from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, ShuffleSplit, GridSearchCV

from sklearn.metrics import roc_curve, auc, precision_score, recall_score

def loadtxtmethod(filename):
  """The function of importing txt text and data using numpy's load method"""
  data = np.loadtxt(filename, dtype=np.float32, delimiter=',')
  return data


def pre_processing(data):
  """The function of pre-processing the data with 
     using Pipeline to assemble all data pre-processing methods"""
  data_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('standardScaler', StandardScaler()),
	('normalize', Normalizer())
  ])
  #train the data transform methods
  data_pipeline.fit(data[:,0:-1])
  #transform the data 
  data[:,0:-1] = data_pipeline.transform(data[:,0:-1])
  return data


def process_ckd(path):
  """The function is used to import the chronic_kidney_disease dataset
  """
  nodataval = arff.loadarff(path)

  # Read the data
  ckd = pd.DataFrame(nodataval[0])
  num_samples, num_features = ckd.shape

  # Decode strings
  is_str_cols = ckd.dtypes == object 
  str_columns = ckd.columns[is_str_cols]
  ckd[str_columns] = ckd[str_columns].apply(lambda s:s.str.decode('utf8'), axis=0)

  # Handle nodata values
  ckd = ckd.replace('?',np.nan)

  # Convert remaining false string columns
  other_numeric_columns = ["sg", "al", "su"] 
  ckd[other_numeric_columns] = ckd[other_numeric_columns].apply(lambda s:pd.to_numeric(s,), axis = 0) 

  # Use categorical data type
  categoric_columns = ckd.columns[ckd.dtypes == "object"]
  ckd[categoric_columns] = ckd[categoric_columns].astype("category")

  #store the features and labels Separately
  y = ckd['class']  
  ckd = ckd.drop(columns=['class'])
  
  #standard the one-hot code and replace the lost data using the mean of data
  fillna_mean_cols = ckd.columns[ckd.dtypes == "float64"]  
  fillna_most_cols = ckd.columns[ckd.dtypes == "category"]
  ckd[fillna_mean_cols] = ckd[fillna_mean_cols].fillna(np.mean(ckd[fillna_mean_cols]))
  ckd[fillna_most_cols] = ckd[fillna_most_cols].fillna(ckd[fillna_most_cols].mode().iloc[0])
  ckd = pd.get_dummies(ckd, drop_first = True)
  
  #transform the string labels to numeric
  trans_table = {"ckd": 1, "notckd": 0} 

  #transform the types of data to numpy array
  y = np.expand_dims(y.map(trans_table).to_numpy(), axis=1)
  X = ckd.to_numpy()

  data2 = np.concatenate((X,y),1)

  #restore the transformed data into the txt document
  np.savetxt('./chronic_kidney_disease.txt', data2, fmt='%f', delimiter=',')


class Logistic_Regression:
  """The method implementing the logistic regression"""
  def __init__(self, input_size):
    #Initialize the weights array and the parameters
    self.dimention = input_size+1
    self.w = np.ones(input_size+1)
    self.alpha = 0.2

  #The function sigmoid
  def f_sigmoid(self, x):
    return 1/(1+np.exp(-x))

  #Define cross-entropy loss
  def loss_function(self, x, y, w):
    if self.f_sigmoid(np.dot(w, x)) == np.zeros(1) or self.f_sigmoid(np.dot(w, x)) == np.ones(1):
      return 0
    return y*np.log(self.f_sigmoid(np.dot(w, x)))+(1-y)*np.log(1-self.f_sigmoid(np.dot(w, x)))

  #use the gradient descent algrithm to optimise the weights in n times
  def classification(self, x, y, n, alpha):
    times = 0
    m = x.shape[0]
    self.alpha = alpha
    loss = []
    #the training iter
    while times<n:
      times += 1
      gradient = 0
      s_loss = 0
      #begin the training for every training data
      for i in zip(x,y):
        s = np.concatenate((i[0],np.ones(1)), 0)
        gradient += (self.f_sigmoid(np.dot(self.w, s))-i[1])*s
        s_loss += self.loss_function(s, i[1], self.w)
      self.w -= self.alpha*gradient/m
      loss.append(-s_loss/m)
    #plot the loss with the training iters
    plt.plot(range(0,n),loss)
    plt.xlabel('the iter')                            
    plt.ylabel('the loss')                             
    plt.title('the loss of training the logistic regression model')
    plt.show()
    return self.w

  #predict the probability for each data to the two classes
  def predict_proba(self, x=None):
    if x is None:
        x = self.x
    x = np.concatenate((x,np.ones((x.shape[0],1))), 1)
    y_pred = self.f_sigmoid(np.dot(x, self.w))
    return y_pred
  
  #predict the labels for every input test data
  def predict(self, x=None):
    if x is None:
        x = self.x
    x = np.concatenate((x,np.ones((x.shape[0],1))), 1)
    y_pred_proba = self.f_sigmoid(np.dot(x, self.w))
    #divide the data to the class with the bigest probability
    y_pred = np.array([0 if y_pred_proba[i] < 0.5 else 1 for i in range(len(y_pred_proba))])
    return y_pred
  
  #calculate the accuracy of the model
  def score(self, y_true=None, y_pred=None):
    if y_true is None or y_pred is None:
        y_true = self.y
        y_pred = self.predict()
    acc = np.mean([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])
    return acc


class Decision_tree:
  """
  The method implementing decision tree"""
  def __init__(self, min_alpha = 0.01, max_alpha = 0.03, step = 0.001):
    #Initialize the parameters
    self.min_alpha = min_alpha
    self.max_alpha = max_alpha
    self.step = step
    self.depth = 3
    self.ccp_alpha = 0.015

  #using the cross validation method to search the best depth of the decision tree
  def search_depth(self, n_depth, X, y):
    depths = np.linspace(1, n_depth, n_depth)
    #define the train data split to validation data
    cvp = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state=42)
    tab_acc_tree = np.zeros(n_depth)
    for i in range(n_depth):
      cal_tree = DecisionTreeClassifier(max_depth=int(depths[i]))
      tab_acc_tree[i] = np.median(np.sqrt(cross_val_score(cal_tree, X, y, scoring='accuracy', cv=cvp)))
    self.depth = int(np.argmax(tab_acc_tree)+1)

  #Explore the appropriate pruning factor for the dataset
  def search_alpha(self, X, y):
    alphas = np.arange(self.min_alpha,self.max_alpha,self.step)
    #define the train data split to validation data
    cvp = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state=42)
    n = int((self.max_alpha-self.min_alpha)/self.step)
    tab_acc_tree = np.zeros(n)
    for i in range(n):
      cal_tree = DecisionTreeClassifier(max_depth=self.depth, ccp_alpha = float(alphas[i]), random_state=42)
      tab_acc_tree[i] = np.median(np.sqrt(cross_val_score(cal_tree, X, y, scoring='accuracy', cv=cvp)))
    self.ccp_alpha = float(np.argmax(tab_acc_tree))

  def classification(self, X_train, y_train, X_test, y_test):
    # K-fold methods for training
    strKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cvscores = []
    cal_trees = []
    for train, validation in strKFold.split(X_train, y_train):
      # Fit the model
      cal_tree = DecisionTreeClassifier(max_depth=self.depth, ccp_alpha=self.ccp_alpha, random_state=42)
      cal_tree.fit(X_train[train], y_train[train])
      cal_trees.append(cal_tree)
      # evaluate the model
      score = cal_tree.score(X_train[validation], y_train[validation])
      cvscores.append(score)
    cal_tree_best = cal_trees[cvscores.index(max(cvscores))]
    #predict the results(labels and probability) using the model
    y_pred = cal_tree_best.predict(X_test)
    y_pred_proba = cal_tree_best.predict_proba(X_test)
    return y_pred, y_pred_proba, np.mean(cvscores)


class Gaussian_NB:
  """The method implementing Gaussian_NB algrithm
  """
  def __init__(self):
    #Initialize the parameters
      self.mean0,self.mean1,self.p_c1 = 0,0,0
      self.var0,self.var1 = 1,1
      self.p0,self.p1 = [0],[0]

    #train the Gaussian_NB method
  def fit(self,trainMatrix,trainCategory):
      numTrainData=len(trainMatrix)
      numFeatures=len(trainMatrix[0])
      self.p_c1=sum(trainCategory)/float(numTrainData)
      #calculate the mean and variance of each classes of training data
      self.mean0 = np.mean(trainMatrix[trainCategory==0],axis = 0)
      self.mean1 = np.mean(trainMatrix[trainCategory==1],axis = 0)
      self.var0 = np.var(trainMatrix[trainCategory==0],axis = 0)
      self.var1 = np.var(trainMatrix[trainCategory==1],axis = 0)
      return self

    #the function to calculate the probability of each test data for the two classes
  def _get_proba(self,testMatrix):
      p0Vect = ((2*np.pi*self.var0)**0.5)*np.exp(-(testMatrix-self.mean0)**2/(2*self.var0**2))
      p1Vect = ((2*np.pi*self.var1)**0.5)*np.exp(-(testMatrix-self.mean1)**2/(2*self.var1**2))
      p_condition0 = reduce(operator.mul, p0Vect.T)
      p_condition1 = reduce(operator.mul, p1Vect.T)
      self.p0 = p_condition0*(1-self.p_c1)
      self.p1 = p_condition1*self.p_c1

    #predict the labels for the test dataset 
  def predict(self,testMatrix):
      self._get_proba(testMatrix)
      label = np.zeros(len(self.p1))
      for i in range(len(label)):
            label[i] = 0 if self.p0[i]>self.p1[i] else 1 
      return label.reshape([-1,1])
    
    #predict the probability for every classes in the test dataset
  def predict_proba(self,testMatrix):
        self._get_proba(testMatrix)
        return np.array([self.p0,self.p1]).T

    #predict the accuracy of the model
  def score(self,y_true, y_pred):
        acc = np.mean([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])
        return acc

class logistic_regression_nn(nn.Module):
    """The method implementing MLP algrithm
  """
    # class initialization
    def __init__(self, input_size, output_size):
      super(logistic_regression_nn, self).__init__()
      self.hid1 = nn.Linear(input_size, 10)  # 8-(10-10)-1
      self.hid2 = nn.Linear(10, 10)
      self.oupt = nn.Linear(10, output_size)

      nn.init.xavier_uniform_(self.hid1.weight) 
      nn.init.zeros_(self.hid1.bias)
      nn.init.xavier_uniform_(self.hid2.weight) 
      nn.init.zeros_(self.hid2.bias)
      nn.init.xavier_uniform_(self.oupt.weight) 
      nn.init.zeros_(self.oupt.bias)
        
    # function to apply the neural network
    def forward(self, x):
      z = torch.tanh(self.hid1(x))
      z = torch.tanh(self.hid2(z))
      z = torch.sigmoid(self.oupt(z))  # for BCELoss()
      return z
    
from sklearn.svm import NuSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

class SVM:
    """
    Use NuSVC from the sklearn Library to classify the data
    """
    def get_parameters(self,trainMatrix,trainCategory):
        """
        Parameters
        ----------
        trainMatrix: Features of the training data set
                The size of trainMatrix is (n,p), where n the number of samples, p the number of
                features.
        trainCategory: Labels of the training data set
                The size of trainCategory is (n,),  where n the number of samples
        """
        param_grid = {'nu': np.linspace(0.1, 0.7, 7), 'gamma': np.linspace(0.01, 1, 10)}
        gsearch = GridSearchCV(NuSVC(), param_grid=param_grid, scoring='accuracy', cv=10)
        gsearch.fit(trainMatrix, trainCategory.ravel())
        self.gamma = gsearch.best_params_['gamma']
        self.nu = gsearch.best_params_['nu']
        return
    
    def classification(self, X_train, y_train, X_test):
        """
        Parameters
        ----------
        X_train: Features of the training data set
                The size of X_train is (n,p), where n the number of samples, p the number of
                features.
        y_train: Labels of the training data set
                The size of y_train is (n,),  where n the number of samples
        X_test: Features of the testing data set
                The size of X_test is (m,p), where m the number of samples, p the number of
                features.
        """
        y_train = y_train.ravel()
        self.get_parameters(X_train,y_train)
        svm = NuSVC(nu = self.nu, gamma = self.gamma,probability=True)
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        y_pred_proba = svm.predict_proba(X_test)
        return y_pred, y_pred_proba
    
    def score(self, y_true, y_pred):
        acc = np.mean([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])
        return acc

def test_logistic(data, dim, n, learning_rate):
    """The test function for the logistic method on the dataset"""
    X_train, X_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:,-1], test_size=0.25, random_state=42)
    #define the model of Logistic_Regression
    LR_model = Logistic_Regression(dim)
    #train the model
    LR_model.classification(X_train, y_train, n, learning_rate)
    #calculate the ratios of evaluating the methods
    y_pred = LR_model.predict(X_test)
    y_score = LR_model.predict_proba(X_test)
    acc_LR = LR_model.score(y_test, y_pred)
    P_score = precision_score(y_test,y_pred)
    R_score = recall_score(y_test,y_pred)
    fpr, tpr, thresholds = roc_curve(y_test,y_score)
    auc1 = auc(fpr, tpr)
    print("The accuracy of the logistic regression model is: ", acc_LR, "\n",
            "The precision of the logistic regression model is: ", P_score, "\n",
            "The recall figure of the logistic regression model is: ", R_score, "\n",
            "The AUC of the logistic regression model is: ", auc1, "\n")
    #plot the ROC curve
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% auc1)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.xlabel('False Positive Rate')                            
    plt.ylabel('True Positive Rate')                             
    plt.title('The ROC curve of the logistic regression model')
    plt.show()

def test_decision_tree(data, max_depth):
    """The test function for the logistic method on the dataset"""
    #here we reduce the original data features with 25 dimensions to 23 dimensions
    if(data.shape[1]>20):
        pca = PCA(n_components=23)
        newX = pca.fit_transform(data[:,0:-1])
    else:
        newX = data[:,0:-1]
    X_train, X_test, y_train, y_test = train_test_split(newX, data[:,-1], test_size=0.25, random_state=42)
    #define the tree model
    Tree_model = Decision_tree()
    #search the optimal depth from 1 to max_depth
    Tree_model.search_depth(max_depth, X_train, y_train)
    #search the optimal alpha from min_alpha to max_alpha
    Tree_model.search_alpha(X_train, y_train)
    #train the model and predict the labels and probability of the test data
    y_pred, y_pred_proba, score = Tree_model.classification(X_train, y_train, X_test, y_test)
    #calculate the ratios of evaluating the method
    P_score = precision_score(y_test,y_pred)
    R_score = recall_score(y_test,y_pred)
    fpr, tpr, thresholds = roc_curve(y_test,y_pred_proba[:,1])
    auc2 = auc(fpr, tpr)
    print("The accuracy of the decision tree model is: ", score, "\n",
            "The precision of the decision tree model is: ", P_score, "\n",
            "The recall figure of the decision tree model is: ", R_score, "\n",
            "The AUC of the decision tree model is: ", auc2, "\n")
    #plot the loss curve
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% auc2)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.xlabel('False Positive Rate')                            
    plt.ylabel('True Positive Rate')                             
    plt.title('The ROC curve of the decision tree model')
    plt.show()

def test_Gaussian_NB(data):
    """The test function for the Gaussian_NB method on the dataset"""
    if(data.shape[1]>20):
        pca = PCA(n_components=5)
        newX = pca.fit_transform(data[:,0:-1])
    else:
        newX = data[:,0:-1]
    X_train, X_test, y_train, y_test = train_test_split(newX, data[:,-1], test_size=0.20, random_state=42)
    #define the model
    gnb = Gaussian_NB()
    #train the model on the training dataset and get the predicted labels
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    #get the accuracy of the model
    acc_gnb = gnb.score(y_test, y_pred)
    #calculate the radios to evaluate the method
    P_score = precision_score(y_test,y_pred)
    R_score = recall_score(y_test,y_pred)
    y_score = (gnb.predict_proba(X_test))[:,1]
    fpr, tpr, thresholds = roc_curve(y_test,y_score)
    Area_Under_Curve = auc(fpr, tpr)
    print("The accuracy of the Gaussian Naive Bayes classifier is: ", acc_gnb, "\n",
        "The precision of the Gaussian Naive Bayes classifier is: ", P_score, "\n",
        "The recall of the Gaussian Naive Bayes classifier is: ", R_score, "\n",
        "The AUC of the Gaussian Naive Bayes classifier is: ", Area_Under_Curve, "\n")
    #plot the ROC curve
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% Area_Under_Curve)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.xlabel('False Positive Rate')                            
    plt.ylabel('True Positive Rate')                             
    plt.title('The ROC curve of the Gaussian Naive Bayes classifier')
    plt.show()


def testMLP(data,epoch,learningrate):
  data = torch.tensor(data)
  X_train, X_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:,-1].reshape(data[:,-1].shape[0],1), test_size=0.25, random_state=42)
  logistic_regression_model = logistic_regression_nn(X_train.shape[1], 1)
  criterion = nn.BCELoss()
  optimizer = torch.optim.SGD(logistic_regression_model.parameters(), lr = learningrate)
  epochs = epoch # number of epochs
  losses = [] # list to stock the loss at each iteration

  # Loop on epochs
  for i in range(epochs):
      
      # compute the prediction using the previous parameters of the neural network
      y_pred = logistic_regression_model.forward(X_train)
      
      # compute and stock the loss
      loss = criterion(y_pred, y_train)
      losses.append(loss.detach().numpy())
      
      # initialize the gradient to zero
      optimizer.zero_grad()
      
      # compute the gradient by back propagation
      loss.backward()
      
      # update the parameter values using the gradient
      optimizer.step()

  ytestpred = logistic_regression_model.forward(X_test)
  yhat = torch.round(ytestpred)
  train_acc = torch.sum(yhat == y_test)
  final_train_acc = train_acc//yhat.shape[0]
  P_score = precision_score(y_test.detach().numpy(),yhat.detach().numpy())
  R_score = recall_score(y_test.detach().numpy(),yhat.detach().numpy())
  fpr, tpr, thresholds = roc_curve(y_test.detach().numpy(),ytestpred.detach().numpy())
  Area_Under_Curve = auc(fpr, tpr)

  print("The accuracy of the MLP model is: ", final_train_acc, "\n",
      "The precision of the MLP model is: ", P_score, "\n",
      "The recall figure of the MLP model is: ", R_score, "\n",
      "The AUC of the MLP model is: ", Area_Under_Curve, "\n")
  plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% Area_Under_Curve)
  plt.legend(loc='lower right')
  plt.plot([0,1],[0,1],'r--')
  plt.xlim([-0.1,1.1])
  plt.ylim([-0.1,1.1])
  plt.xlabel('False Positive Rate')                            
  plt.ylabel('True Positive Rate')                             
  plt.title('The ROC curve of the decision tree model')
  plt.show()

def testSVM(data):
    """
    The test function for the SVM method on the dataset
    
    Input
    ----------
    data: the dataset
        The size of input is (n,p+1), where n is the number of data, p the number of features. The latest colone is the the label.
    """
    X_train, X_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:,-1].reshape(data[:,-1].shape[0],1), test_size=0.25, random_state=42)
    #define the model
    svm = SVM()
    #train the model on the training dataset and get the predicted labels
    y_pred, y_pred_proba = svm.classification(X_train, y_train, X_test)
    #get the accuracy of the model
    acc_svm = svm.score(y_test,y_pred)
    #calculate the radios to evaluate the method
    P_score = precision_score(y_test,y_pred)
    R_score = recall_score(y_test,y_pred)
    y_score = y_pred_proba[:,1]
    fpr, tpr, thresholds = roc_curve(y_test,y_score)
    Area_Under_Curve = auc(fpr, tpr)
    print("The accuracy of the SVM classifier is: ", acc_svm, "\n",
    "The precision of the SVM classifier is: ", P_score, "\n",
    "The recall of the SVM classifier is: ", R_score, "\n",
    "The AUC of the SVM classifier is: ", Area_Under_Curve, "\n")
    #plot the ROC curve
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% Area_Under_Curve)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.xlabel('False Positive Rate')                            
    plt.ylabel('True Positive Rate')                             
    plt.title('The ROC curve of the SVM classifier')
    plt.show()
