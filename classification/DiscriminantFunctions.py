from ml import utilities
import numpy as np
import sys
from abc import ABC,abstractmethod

class ML(ABC):
    coef_ = []
    classes_ = []
    intercept_ = []
    
    @abstractmethod
    def fit(self,X,y):
        pass
    
    @abstractmethod
    def predict(self,X):
        pass
    
    @abstractmethod
    def score(self,X,y):
        pass
    

class FisherLinearDiscriminant(ML):
    means_ = []
    
    def fit(self,X,y):
        '''
        X:2d array
            this contians the features of each sample [[f1,f2,..fn],[f1,f2,..fn]]
        y:1d array
            the label of each sample on X  
        '''
        self.classes_ = list(set(y))
        sum_samples = dict.fromkeys(self.classes_, [0]* len(X[0]))
        n_samples = dict.fromkeys(self.classes_, 0)
        for index, value in enumerate(y):
            sum_samples[value] = [old_val + new_val for old_val, new_val in zip(sum_samples[value], X[index])]
            n_samples[value] += 1
        for index, value in enumerate(self.classes_): #calculate means of each class
            self.means_.append(np.divide(sum_samples[value],n_samples[value]))
        
        for class_index,class_ in enumerate(self.classes_):
            rest_means = [0]*len(X[0])
            rest_sample_size = 0
            for index, value in enumerate(self.classes_): #calculate means of each class
                if value != class_:
                    rest_means =np.add(rest_means,sum_samples[value])
                    rest_sample_size += 1
            rest_means = np.divide(rest_means,rest_sample_size)
            class_variance = [[0]*len(X[0]) for i in range(len(X[0]))]
            rest_classes_variance = [[0]*len(X[0]) for i in range(len(X[0]))]
            for i, j in enumerate(y):
                if j == class_:
                    class_variance = np.add(class_variance,self._sample_variance(self.means_[class_index],X[i]))
                else:
                    rest_classes_variance = np.add(rest_classes_variance,self._sample_variance(rest_means,X[i]))
            sw =np.add(class_variance,rest_classes_variance)
            sw_inverse = np.linalg.pinv(sw)
            sub_means = np.subtract(rest_means,self.means_[class_index])
            wieghts = np.matmul(sw_inverse,utilities.transpose_matrix(sub_means))
            self.coef_.append(np.transpose(wieghts)[0])
            avg_means = np.divide(np.add(rest_means,self.means_[class_index]),2)
            self.intercept_.append(np.multiply(np.dot(self.coef_[class_index],avg_means),-1))
            
    def predict(self,X):
        '''
        X:2d array
            this contians the features of each sample [[f1,f2,..fn],[f1,f2,..fn]]
        
        return :
            y:1d array
                predicted label of each sample on X  
        '''
        y = []
        for features in X:
            predicted_y = sys.maxsize
            predicted_class_index = -1
            for index,coef in enumerate(self.coef_):
                tmp_predicted_y =np.dot(coef,features) + self.intercept_[index]
                if predicted_y > tmp_predicted_y:
                    predicted_y = tmp_predicted_y
                    predicted_class_index = index
            y.append(self.classes_[predicted_class_index])
            
        return y
        
    
    def score(self,X,y):
        '''
        X:2d array
            this contians the features of each sample [[f1,f2,..fn],[f1,f2,..fn]]
        y:1d array
            the label of each sample on X  
        
        return: score , confusion matrics
            score: float
                score of correct simples
            confustion matrics:2d array
                            contians wrong and right predictin for each class
        '''
    
        confusion_matrix = [[0]*len(self.classes_) for i in range(len(self.classes_))]
        correct_predict = 0
        for real_class_index,features in enumerate(X):
            predicted_y = sys.maxsize
 
            predicted_class_index = -1
            for index,coef in enumerate(self.coef_):
                tmp_predicted_y =np.dot(coef,features) + self.intercept_[index]
                if predicted_y > tmp_predicted_y:
                    predicted_y = tmp_predicted_y
                    predicted_class_index = index
            y_index = self.classes_.index(y[real_class_index])
            if y_index == predicted_class_index:
                correct_predict += 1
            confusion_matrix[y_index][predicted_class_index] += 1
        return float(correct_predict/len(X)) , confusion_matrix
    
    def _sample_variance(self,class_mean,sample):
        sample_sub_mean_transpose = np.subtract(sample,class_mean)
        sample_sub_mean =np.transpose([sample_sub_mean_transpose])
        return np.matmul(sample_sub_mean,[sample_sub_mean_transpose])

class LeastSquares(ML):
    def fit(self,X,y):
        pass

    def score(self,X,y):
        pass
    
    def predict(self,X):
        pass
