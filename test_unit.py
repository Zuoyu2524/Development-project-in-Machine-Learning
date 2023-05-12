import unittest
import Fonctions

class mytest(unittest.TestCase): 
  @classmethod
  def setUpClass(cls): 
    #Read the data from the txt document
    Fonctions.process_ckd("./chronic_kidney_disease_full.arff") 

  @classmethod
  def setUp(self):
    data_ckd = Fonctions.loadtxtmethod("./chronic_kidney_disease.txt")
    data_bank = Fonctions.loadtxtmethod("./data_banknote_authentication.txt")
    self.data1 = Fonctions.pre_processing(data_ckd)
    self.data2 = Fonctions.pre_processing(data_bank)
  
  @classmethod
  def test_logistic(self):
    print("The test of logistic regression model for chronic_kidney_disease")
    Fonctions.test_logistic(self.data1, self.data1.shape[1]-1, 4000, 0.01)
    print("The test of logistic regression model for banknote_authentication_dataset")
    Fonctions.test_logistic(self.data2, self.data2.shape[1]-1, 2500, 0.01)
  
  @classmethod
  def test_decision_tree(self):
    print("The test of decision tree model for chronic_kidney_disease")
    Fonctions.test_decision_tree(self.data1, 20)
    print("The test of decision tree model for banknote_authentication_dataset")
    Fonctions.test_decision_tree(self.data2, 10)

  @classmethod
  def test_Gaussian_NB(self):
    print("The test of gaussian naive bayes model for chronic_kidney_disease")
    Fonctions.test_Gaussian_NB(self.data1)
    print("The test of gaussian naive bayes model for banknote_authentication_dataset")
    Fonctions.test_Gaussian_NB(self.data2)

  @classmethod
  def test_MLP(self):
    print("The test of mlp model for chronic_kidney_disease")
    Fonctions.testMLP(self.data1, 1000, 0.01)
    print("The test of mlp model for banknote_authentication_dataset")
    Fonctions.testMLP(self.data2, 1000, 0.01)

  @classmethod
  def test_SVM(self):
    print("The test of SVM model for chronic_kidney_disease")
    Fonctions.testSVM(self.data1)
    print("The test of SVM model for banknote_authentication_dataset")
    Fonctions.testSVM(self.data2)

if __name__ == '__main__':
    unittest.main(verbosity=2)
