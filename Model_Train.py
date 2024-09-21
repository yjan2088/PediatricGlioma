#%%
import torch
import random
import numpy as np
import os
import pickle

def seed_everything(seed=42):
    """"
    Seed everything.
    """   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
SEED = 2023 #random.randint(0,100000)
seed_everything(SEED)

np.seterr(divide='ignore', invalid='ignore') #Ignore warning: #RuntimeWarning: invalid value encountered in true_divide


import pandas as pd
dataset = pd.read_csv('./new_data/332.csv')
dataset_train = pd.read_csv('./new_data/20230823_train2.csv')
dataset_test = pd.read_csv('./new_data/20230823_test2.csv')

dataset.columns = [name.replace('.',' ') for name in dataset.columns]
dataset_train.columns = [name.replace('.',' ') for name in dataset_train.columns]
dataset_test.columns = [name.replace('.',' ') for name in dataset_test.columns]

time_column = 'Survival months'
event_column = 'Status'
features = ['Age', 'Gender', 'Race','Histological type','Stage',  
            'Laterality','Location of the tumor', 'Tumor size',
            'Extension',"Surgery",'Radiation','Chemotherapy', ]
# features = np.setdiff1d(dataset.columns, [time_column, event_column]).tolist()
# dataset.head()

cox_features = features[:]
cox_features.remove('Gender')
cox_features.remove('Stage')
import numpy as np

# Creating the X, T and E inputs
X_train, X_test = dataset_train[features], dataset_test[features]
T_train, T_test = dataset_train[time_column], dataset_test[time_column]
E_train, E_test = dataset_train[event_column], dataset_test[event_column]

cox_X_train, cox_X_test = dataset_train[cox_features], dataset_test[cox_features]
cox_T_train, cox_T_test = dataset_train[time_column], dataset_test[time_column]
cox_E_train, cox_E_test = dataset_train[event_column], dataset_test[event_column]


#%%
from pysurvival.utils.metrics import concordance_index
from sklearn.model_selection import RepeatedKFold
from pysurvival.models.survival_forest import ConditionalSurvivalForestModel
class BaseFun:
    def parse_param(self, param):
        for key in param:
            if isinstance(param[key], str):
                param[key] = '"{}"'.format(param[key])
        return ','.join(['{} = {}'.format(key, param[key]) for key in param])
    def get_random_param(self,space):
        param = {}
        for key in space:
            if  key == 'structure':
                items = []
                for i in range(1,random.choice(space['structure']['num_layers'])+1):
                    items.append(
                        {
                        'activation': random.choice(space['structure']['activations']),
                        'num_units':random.choice(space['structure']['num_units'])
                        }
                    )
                param['structure'] = items
            else:
                param[key] = random.choice(space[key])
        return param
    
    def tuning_and_construct(self,X, T, E,max_iter=100):
        self.tuning_result = self.tuning_with_space(X, T, E,self.space,max_iter=max_iter)
        self.model = self.fit_model(X, T, E,**self.tuning_result['best_param'])
        return self.model
    def tuning_with_space(self,x,t,e,space,max_iter=100):
        [x,t,e] = [item if isinstance(item, np.ndarray) else np.array(item) for item in [x,t,e]]
        scores = []
        best_score = 0
        best_param = {}
        num = 1
        while True:
            param = self.get_random_param(space)
            print(param)
            print('Number {} iteration'.format(num), end=' ... ')
            # split train data to 5 parts
            rkf = RepeatedKFold(n_splits=5, n_repeats=1)
            score_iter = []
            for train_index, test_index in rkf.split(x):
                x_train, x_test = x[train_index], x[test_index]
                t_train, t_test = t[train_index], t[test_index]
                e_train, e_test = e[train_index], e[test_index]
                try:
                    model = self.fit_model(x_train, t_train, e_train,**param)
                    score = concordance_index(model, x_test, t_test, e_test)
                except Exception as err:
                    print(str(err))
                    break
                score_iter.append(score)
            mean_score = np.mean(score_iter)
            print('mean_c_index: ', mean_score, end=' ')
            if mean_score > best_score:
                best_score = mean_score
                best_param = param
                print('is the best so far')
            else:
                print('')
            scores.append({'iteration':num,'mean_score': mean_score, 'param': param})
            if num == max_iter:
                break
            num += 1
        return {'best_param': best_param, 'best_score': best_score, 'scores': scores}


from pysurvival.models.multi_task import NeuralMultiTaskModel
class NMTLR(BaseFun):
    def __init__(self):
        self.space = {
            'structure': {'num_layers':[1,2,3,4,5],
                          'num_units': [i for i in range(8, 100)],
                          'activations': ["Atan", "BentIdentity", "BipolarSigmoid", "CosReLU", 
                                           "Gaussian", "Hardtanh", "InverseSqrt", "LeakyReLU", 
                                          "LeCunTanh", "LogLog", "LogSigmoid", "ReLU", "SELU", "Sigmoid", 
                                          "Sinc", "SinReLU","Softmax", "Softplus", "Softsign", "Swish", "Tanh"]
                         },
            'optimizer':['adadelta','adagrad','adam','adamax','rmsprop','sgd'],
            'bins' : [i for i in range(10,100)],
            'lr': [round(1e-5 * i, 5) for i in range(1, 100 + 1)],
            'num_epochs': [i for i in range(50, 1000 + 1)],
            'dropout': [round(0.1 * i, 2) for i in range(1, 4 + 1)],
            'l2_reg': [round(0.0001 * i, 5) for i in range(1, 100 + 1)],
            'l2_smooth':[round(0.0001 * i, 5) for i in range(1, 100 + 1)],
            'batch_normalization' : [False, True]
        }
        self.model = None
    def fit_model(self,X, T, E,**kwargs):
        structure = [{'activation': 'ReLU', 'num_units': 128}]
        bins = 100
        if 'structure' in kwargs:
            structure = kwargs['structure']
            del kwargs['structure']
        if 'bins' in kwargs:
            bins = kwargs['bins']
            del kwargs['bins']
        self.model = NeuralMultiTaskModel(structure=structure,bins=bins)
        eval('self.model.fit(X, T, E,{})'.format(self.parse_param(kwargs)))
        return self.model

from pysurvival.models.semi_parametric import NonLinearCoxPHModel
class DeepSurv(BaseFun):
    def __init__(self):
        self.space = {
            'structure': {'num_layers':[1,2,3,4,5],
                          'num_units': [i for i in range(8, 100)],
                          'activations': ["Atan", "BentIdentity", "BipolarSigmoid", "CosReLU", 
                                           "Gaussian", "Hardtanh", "InverseSqrt", "LeakyReLU", 
                                          "LeCunTanh", "LogLog", "LogSigmoid", "ReLU", "SELU", "Sigmoid", 
                                          "Sinc", "SinReLU","Softmax", "Softplus", "Softsign", "Swish", "Tanh"]
                         },
            'optimizer':['adadelta','adagrad','adam','adamax','rmsprop','sgd'],
            'lr': [round(1e-5 * i, 5) for i in range(1, 100 + 1)],
            'num_epochs': [i for i in range(50, 5000 + 1)],
            'dropout': [round(0.1 * i, 2) for i in range(1, 4 + 1)],
            'l2_reg': [round(0.0001 * i, 5) for i in range(1, 100 + 1)],
            'batch_normalization' : [False, True]
        }
        self.model = None
    def fit_model(self,X, T, E,**kwargs):
        structure = [{'activation': 'ReLU', 'num_units': 128}]
        bins = 100
        if 'structure' in kwargs:
            structure = kwargs['structure']
            del kwargs['structure']
        self.model = NonLinearCoxPHModel(structure=structure)
        eval('self.model.fit(X, T, E,{})'.format(self.parse_param(kwargs)))
        return self.model


from pysurvival.models.survival_forest import RandomSurvivalForestModel
class RSF(BaseFun):
    def __init__(self):
        self.space = {
            'num_trees': [i for i in range(20, 1000 + 1)],
            'max_features': ['sqrt', 'log2', 'all', 0.1, 0.2],
            'min_node_size': [i for i in range(5, 80 + 1)],
            'sample_size_pct': [round(0.2 * i, 2) for i in range(1, 4 + 1)],
            'importance_mode': ['impurity', 'impurity_corrected', 'permutation', 'normalized_permutation']
        }
        self.model = None
    def fit_model(self,X, T, E,**kwargs):
        if 'num_trees' in kwargs:
            self.model = RandomSurvivalForestModel(num_trees=kwargs['num_trees'])
            del kwargs['num_trees']
        else:
            self.model = ConditionalSurvivalForestModel()
        eval('self.model.fit(X, T, E,seed=SEED,{})'.format(self.parse_param(kwargs)))
        return self.model

#%% 
# from pysurvival.models.semi_parametric import CoxPHModel
# def cph(X_train, T_train, E_train):
#     model = CoxPHModel()
#     model.fit(X_train, T_train, E_train, lr=0.2, l2_reg=0.01)
#     return model

# cph_model = cph(cox_X_train, cox_T_train, cox_E_train)
# c_index_train = concordance_index(cph_model, cox_X_train, cox_T_train, cox_E_train)
# c_index_test = concordance_index(cph_model, cox_X_test, cox_T_test, cox_E_test)

# print('C-index of train: {:.4f}; C-index of test: {:.4f}'.format(c_index_train,c_index_test)) #C-index: 0.6996

# from pysurvival.utils import save_model
# save_model(cph_model, './outputModel_1/CoxPH.zip')
# from pysurvival.utils.display import integrated_brier_score
# ibs = integrated_brier_score(cph_model, X_train, T_train, E_train,t_max=None, figure_size=(20, 6.5) )
# print('IBS: {:.3f}'.format(ibs))
#%%
max_iter = 200

nmtlr = NMTLR()
nmtlr.tuning_and_construct(X_train, T_train, E_train,max_iter=max_iter)


# # # NMTLR Result
# # c_index_train = concordance_index(nmtlr.model, X_train, T_train, E_train)
# # c_index_test = concordance_index(nmtlr.model, X_test, T_test, E_test)
# # print('C-index of train: {:.4f}; C-index of test: {:.4f}'.format(c_index_train,c_index_test)) #C-index: 0.6996

# # from pysurvival.utils.display import integrated_brier_score
# # ibs = integrated_brier_score(nmtlr.model, X_test, T_test, E_test,figure_size=(20, 6.5) )
# # print('IBS: {:.3f}'.format(ibs))+

# nmtlr_model = nmtlr.model
# # from pysurvival.utils import save_model
# # save_model(nmtlr.model, './outputModel/nmtlr_model_10.zip')
# import pickle
# with open('./outputModel_1/nmtlr_model_200.pkl', 'wb') as f:
#     pickle.dump(nmtlr_model, f)
#     print('NMTLR Save')
#%%
max_iter = 1000
import time 
start = time.time()
deepsurv = DeepSurv()
deepsurv.tuning_and_construct(X_train, T_train, E_train,max_iter=max_iter)

# # c_index_train = concordance_index(deepsurv.model, X_train, T_train, E_train)
# # c_index_test = concordance_index(deepsurv.model, X_test, T_test, E_test)
# # print('C-index of train: {:.4f}; C-index of test: {:.4f}'.format(c_index_train,c_index_test)) #C-index: 0.6996

# # from pysurvival.utils.display import integrated_brier_score
# # ibs = integrated_brier_score(deepsurv.model, X_test, T_test, E_test, figure_size=(20, 6.5) )
# # print('IBS: {:.3f}'.format(ibs))

# 
deepsurv_model = deepsurv.model
import pickle
with open('./outputModel_1/deepsurv_model_1000.pkl', 'wb') as f:
    pickle.dump(deepsurv_model, f)
    print('DeepSurv Save！')
print(time.time() - start)

#%% RSF
max_iter = 200
rsf = RSF()
rsf.tuning_and_construct(X_train, T_train, E_train,max_iter=max_iter)

rsf_model = rsf.model
from pysurvival.utils import save_model
save_model(rsf_model, './outputModel_1/rsf_model_200.zip')

