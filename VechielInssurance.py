#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense
import keras as k
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_curve, auc ,roc_auc_score,confusion_matrix,accuracy_score, f1_score, precision_score, recall_score,fbeta_score
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from random import randint
import random
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error as mae
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score


# In[15]:


df=pd.read_csv('/media/areen/areen/archive/train.csv')
df.columns


# In[16]:


df


# In[17]:


df=df.drop('id',axis=1)


# In[18]:


df=df.round().drop_duplicates()
df


# In[19]:


df.isnull().sum()


# In[20]:


gender_map = {'Female':1, 'Male':2}
df['Gender'] = df['Gender'].map(gender_map)
df.head()


# In[21]:


vehicle_age_map = {'< 1 Year':1, '1-2 Year':2, '> 2 Years':3}
df['Vehicle_Age'] = df['Vehicle_Age'].map(vehicle_age_map)
df.head()


# In[22]:


vehicle_damage_map = {'Yes':1, 'No':2}
df['Vehicle_Damage'] = df['Vehicle_Damage'].map(vehicle_damage_map)
df.head()


# In[23]:


import seaborn as sns
sns.countplot('Response',data=df).plot()


# In[24]:


Y=df.Response
balance_sample= df.drop(["Response"], axis = 1)
X=balance_sample
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
print(X_train.shape)
print(X_test.shape)


# In[25]:


X


# In[13]:


def population_func(numOfGeneration):
    activation = ['identity','logistic', 'tanh', 'relu']
    solver = ['lbfgs','sgd', 'adam']
    nb_neurons = [512, 768, 1024]
    nb_layers = [3, 4,5, 6]
    
    pop =  np.array([[random.choice(activation), random.choice(solver),random.choice(nb_layers), random.choice(nb_neurons)]])
    for i in range(0, numOfGeneration-1):
        pop = np.append(pop, [[random.choice(activation), random.choice(solver),random.choice(nb_layers), random.choice(nb_neurons)]], axis=0)
    return pop

def crossover_func(p1, p2):
    child = [p1[0], p2[1], p1[2], p2[3]]    
    return child

def mutation_func(child, prob_mutation):
    child_ = np.copy(child)
    for c in range(0, len(child_)):
        if np.random.rand() >= prob_mutation:
            k = randint(2,3)
            child_[c,k] = int(child_[c,k]) + randint(1, 4)
    return child_


def fitness_func(pop, X_train, y_train, X_test, y_test): 
    fitness = []
    for w in pop:
        model = MLPClassifier(learning_rate_init=0.09, activation=w[0],solver = w[1], 
                            alpha=1e-5, hidden_layer_sizes=(int(w[2]), 
                            int(w[3])),  max_iter=100, n_iter_no_change=80)
        try:
            model.fit(X_train, y_train)
            f1_score = f1_score(model.predict(X_test), y_test,average='micro')
            fitness.append([f1_score, model, w])
        except:
            pass
    return fitness


def main_func(X_train, y_train, X_test, y_test, num_epochs = 10, numOfGeneration=10, prob_mutation=0.8):
    pop = population_func(numOfGeneration)
    fitness = fitness_func(pop,  X_train, y_train, X_test, y_test)
    sortedFitness = np.array(list(reversed(sorted(fitness,key=lambda x: x[0]))))

    for j in range(0, num_epochs):
        length = len(sortedFitness)
        parent1 = sortedFitness[:,2][:length//2]
        parent2 = sortedFitness[:,2][length//2:]

        child1 = [crossover_func(parent1[i], parent2[i]) for i in range(0, np.min([len(parent2), len(parent1)]))]
        child2 = [crossover_func(parent2[i], parent1[i]) for i in range(0, np.min([len(parent2), len(parent1)]))]
        child2 = mutation_func(child2, prob_mutation)
        
        fitness_child1 = fitness_func(child1,X_train, y_train, X_test, y_test)
        fitness_child2 = fitness_func(child2, X_train, y_train, X_test, y_test)
        sortedFitness = np.concatenate((sortedFitness, fitness_child1, fitness_child2))
        sort = np.array(list(reversed(sorted(sortedFitness,key=lambda x: x[0]))))
        
        sortedFitness = sort[0:numOfGeneration, :]
        best_model = sort[0][1]
        
    return best_model

 GANN= main_func(X_train, y_train, X_test, y_test, num_epochs = 10, numOfGeneration=10, prob_mutation=0.7)

predictions = GANN.predict(X_test)
print (accuracy_score(predictions, y_test))


# In[14]:


y_pred_proba = GANN.predict_proba(X_test)[:, 1]

#roc 
roc=roc_auc_score(y_test,y_pred_proba)
roc

# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
print('false postive rate',fpr.mean())
print('true  postive rate',tpr.mean())


# In[15]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

#calculating precision and reall
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
 
print('Precision: ',precision)
print('Recall: ',recall)
 
#Plotting Precision-Recall Curve
disp = plot_precision_recall_curve(GANN, X_test, y_test)

average_precision = average_precision_score(y_test, predictions)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))

disp.ax_.set_title('Precision-Recall curve: ''AP={0:0.2f}'.format(average_precision))


# In[16]:


from sklearn.metrics import roc_curve, auc ,roc_auc_score,confusion_matrix,accuracy_score, f1_score, precision_score, recall_score,fbeta_score

#plot roc curve 
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
print('Area under the Receiver Operating Characteristic curve:', roc_auc_score(y_test, y_pred_proba))


# In[22]:


model_results = pd.DataFrame(columns = ['Model', 'Accuracy', 'Precision','Recall', 'F1 Score', 'F2 Score','ROC','TPR','FPR'])


# In[23]:


model_results.loc[len(model_results.index)] = models_results(GANN,"GA-Nueral network",y_test,predictions)
print(model_results)


# In[18]:


def models_results(model,model_name,y_tst,y_pred):
    cnf_matrix = confusion_matrix(y_tst, y_pred)
    print(cnf_matrix)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    #Accuracy
    cross_val_acc = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    acc=round(cross_val_acc.max() * 100 , 2)
    
    #F1
    f1 =cross_val_score(model, X_train, y_train, scoring='f1', cv=cv, n_jobs=-1)

    #precision
    prec =cross_val_score(model, X_train, y_train, scoring='precision', cv=cv, n_jobs=-1)
   
    #recall
    rec =cross_val_score(model, X_train, y_train, scoring='recall', cv=cv, n_jobs=-1)
   
    #F2
    f2 = fbeta_score(y_tst, y_pred, beta=2.0)
    
    #ROC
    roc =cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)

    # TPR
    TPR = TP/(TP+FN)
    
    #FPR
    FPR = FP/(FP+TN)
    
    results=[model_name, acc, prec.max(), rec.max(), f1.max(), f2,roc.max(),TPR.max(),FPR.max()]
    return results


# In[20]:


from sklearn.model_selection import KFold

cv = KFold(n_splits=3, random_state=1, shuffle=True)
cv

