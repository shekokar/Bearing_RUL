import pandas as pd
import pywt
import numpy as np
import matplotlib.pyplot as plt


class GaussianProcess:
    def __init__(self, n_restarts, optimizer):
        self.n_restarts = n_restarts
        self.optimizer = optimizer
        
       
    def Corr(self, X1, X2, theta):
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            K[i,:] = np.exp(-np.sum(theta*(X1[i,:]-X2)**2, axis=1))
            
        return K
 
       
    def Neglikelihood(self, theta):
        theta = 10**theta    # Correlation length
        n = self.X.shape[0]  # Number of training instances
        one = np.ones((n,1))      # Vector of ones
        
        # Construct correlation matrix
        K = self.Corr(self.X, self.X, theta) + np.eye(n)*1e-10
        L = np.linalg.cholesky(K)
        #inv_K = np.linalg.inv(K)   # Inverse of correlation matrix
        
        # Mean estimation
        mu = (one.T @ (cho_solve((L, True), self.y))) / \
            (one.T @ (cho_solve((L, True), one)))
        # mu = (one.T @ inv_K @ self.y)/ (one.T @ inv_K @ one)
        
        # Variance estimation
        SigmaSqr = (self.y-mu*one).T @ (cho_solve((L, True), self.y-mu*one)) / n
        # SigmaSqr = (self.y-mu*one).T @ inv_K @ (self.y-mu*one) / n
        
        # Compute log-likelihood
        LnDetK = 2*np.sum(np.log(np.abs(np.diag(L))))
        # DetK = np.linalg.det(K)
        LnLike = -(n/2)*np.log(SigmaSqr) - 0.5*LnDetK
        
        # Update attributes
        self.K, self.L, self.mu, self.SigmaSqr = K, L, mu, SigmaSqr
        
        return -LnLike.flatten()
        
        
    def fit(self, X, y):
        self.X, self.y = X, y
        lb, ub = -3, 2
        
        # Generate random starting points (Latin Hypercube)
        lhd = lhs(self.X.shape[1], samples=self.n_restarts)
        
        # Scale random samples to the given bounds 
        initial_points = (ub-lb)*lhd + lb
        
        # Create A Bounds instance for optimization
        bnds = Bounds(lb*np.ones(X.shape[1]),ub*np.ones(X.shape[1]))
        
        # Run local optimizer on all points
        opt_para = np.zeros((self.n_restarts, self.X.shape[1]))
        opt_func = np.zeros((self.n_restarts, 1))
        for i in range(self.n_restarts):
            res = minimize(self.Neglikelihood, initial_points[i,:], method=self.optimizer,
                bounds=bnds)
            opt_para[i,:] = res.x
            opt_func[i,:] = res.fun
        
        # Locate the optimum results
        self.theta = opt_para[np.argmin(opt_func)]
        
        # Update attributes
        self.NegLnlike = self.Neglikelihood(self.theta)
        
    
    def predict(self, X_test):        
        n = self.X.shape[0]
        one = np.ones((n,1))
        
        # Construct correlation matrix between test and train data
        k = self.Corr(self.X, X_test, 10**self.theta)
        
        # Mean prediction
        f = self.mu + k.T @ (cho_solve((self.L, True), self.y-self.mu*one))
        # f = self.mu + k.T @ self.inv_K @ (self.y-self.mu*one)
        
        # Variance prediction
        SSqr = self.SigmaSqr*(1 - np.diag(k.T @ (cho_solve((self.L, True), k))))
        # SSqr = self.SigmaSqr*(1 - np.diag(k.T @ self.inv_K @ k))
        
        return f.flatten(), SSqr.flatten()



#PREPROCESSES DATA
def preprocess(x,Q):
    wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric')
    terminal_nodes = wp.get_leaf_nodes(True)
    Edn = np.array([terminal_node.data[0]**2 + terminal_node.data[1]**2 for terminal_node in terminal_nodes])
    Et = 0
    for E in Edn:
        Et = Et + E
    Pdn = []
    for i in range(len(Edn)):
        Pdn.append([terminal_nodes[i].data[0] ** 2 / Edn[i] , terminal_nodes[i].data[1] ** 2 / Edn[i] ])
    Pdn = np.array(Pdn)
    Pt = np.array([ E/Et for E in Edn])
    H = (renyi_entropy(Pt,0.5))
    D = (divergence(Pt,Q,0.5,0.5))
    return H,D

def renyi_entropy(P,alpha):
    P1 = P
    entropy = 1/(1-alpha) * np.log(np.sum(P1**alpha))
    return entropy

def divergence(P,Q,w,alpha):
    d = renyi_entropy(((w*P)+(1-w)*Q) , alpha) - (w*renyi_entropy(P,alpha) + (1-w)*renyi_entropy(Q,alpha))
    return d

def stat_complexity(Q0,P,P0,w,alpha):
    sc = Q0 * divergence(P,P0,w,alpha) * renyi_entropy(P,alpha)
    return sc

def feature_extract(name,size):
    data = pd.read_csv('../dataset/Test_set/'+str(name)+'/acc_00001.csv',header=None)
    h = data[4].to_numpy()
    v = data[5].to_numpy()
    x = np.sqrt(h**2 + v**2)
    wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric')
    terminal_nodes = wp.get_leaf_nodes(True)
    Edn = np.array([terminal_node.data[0]**2 + terminal_node.data[1]**2 for terminal_node in terminal_nodes])
    Et = 0
    for E in Edn:
        Et = Et + E
    Pt = np.array([ E/Et for E in Edn])
    Q = Pt
    H = []
    D = []
    num = ["{0:05}".format(i) for i in range(1,size+1)]
    for n in num:
        data = pd.read_csv('../dataset/Test_set/'+str(name)+'/acc_'+str(n)+'.csv',header=None)
        h = data[4].to_numpy()
        v = data[5].to_numpy()
        x = np.sqrt(h**2 + v**2)
        H1,D1 = preprocess(x,Q)
        H.append(H1)
        D.append(D1)
        print(len(H))
    X = np.array([H,D]).T
    return X

X_train_1 = pd.read_csv('./final_preprocessed_data/bearing1_1.csv',header=None)#Training data for condition 1
X_train_2 = pd.read_csv('./final_preprocessed_data/bearing2_1.csv',header=None)#Training data for condition 2
X_train_3 = pd.read_csv('./final_preprocessed_data/bearing3_1.csv',header=None)#Training data for condition 3

t1 = [i/X_train_1.shape[0] for i in range(1,X_train_1.shape[0]+1)]#Target fn for condition 1
t2 = [i/X_train_2.shape[0] for i in range(1,X_train_2.shape[0]+1)]#Target fn for condition 2
t3 = [i/X_train_3.shape[0] for i in range(1,X_train_3.shape[0]+1)]#Target fn for condition 3

'''print('Preprocessing data for Bearing1_3')
X_test_13 = feature_extract('Bearing1_3',1802) #1802 is acc_xxxxx ie xxxxx of last csv file for Bearing1_3
print('Preprocessing data for Bearing1_4')
X_test_14 = feature_extract('Bearing1_4',188) #188 is acc_xxxxx ie xxxxx of last csv file for Bearing1_4
print('Preprocessing data for Bearing1_5')
X_test_15 = feature_extract('Bearing1_5',383) #383 is acc_xxxxx ie xxxxx of last csv file for Bearing1_5
print('Preprocessing data for Bearing1_6')
X_test_16 = feature_extract('Bearing1_6',383) #383 is acc_xxxxx ie xxxxx of last csv file for Bearing1_6
print('Preprocessing data for Bearing1_7')
X_test_17 = feature_extract('Bearing1_7',250) #250 is acc_xxxxx ie xxxxx of last csv file for Bearing1_7
print('Preprocessing data for Bearing2_3')
X_test_23 = feature_extract('Bearing2_3',1202) #1202 is acc_xxxxx ie xxxxx of last csv file for Bearing2_3
print('Preprocessing data for Bearing2_4')
X_test_24 = feature_extract('Bearing2_4',101) #101 is acc_xxxxx ie xxxxx of last csv file for Bearing2_4
print('Preprocessing data for Bearing2_5')
X_test_25 = feature_extract('Bearing2_5',335) #335 is acc_xxxxx ie xxxxx of last csv file for Bearing2_5
print('Preprocessing data for Bearing2_6')
X_test_26 = feature_extract('Bearing2_6',572) #572 is acc_xxxxx ie xxxxx of last csv file for Bearing2_6
print('Preprocessing data for Bearing2_7')
X_test_27 = feature_extract('Bearing2_7',28) #28 is acc_xxxxx ie xxxxx of last csv file for Bearing2_7
print('Preprocessing data for Bearing3_3')
X_test_33 = feature_extract('Bearing3_3',58) #58 is acc_xxxxx ie xxxxx of last csv file for Bearing3_3'''

#loading feature extracted data
print('Preprocessing data for Bearing1_3')
X_test_13 = pd.read_csv('bearing1_3.csv',header=None) #1802 is acc_xxxxx ie xxxxx of last csv file for Bearing1_3
print('Preprocessing data for Bearing1_4')
X_test_14 = pd.read_csv('bearing1_4.csv',header=None) #1802 is acc_xxxxx ie xxxxx of last csv file for Bearing1_3
print('Preprocessing data for Bearing1_5')
X_test_15 = pd.read_csv('bearing1_5.csv',header=None) #1802 is acc_xxxxx ie xxxxx of last csv file for Bearing1_3
print('Preprocessing data for Bearing1_6')
X_test_16 = pd.read_csv('bearing1_6.csv',header=None) #1802 is acc_xxxxx ie xxxxx of last csv file for Bearing1_3
print('Preprocessing data for Bearing1_7')
X_test_17 = pd.read_csv('bearing1_7.csv',header=None) #1802 is acc_xxxxx ie xxxxx of last csv file for Bearing1_3
print('Preprocessing data for Bearing2_3')
X_test_23 = pd.read_csv('bearing2_3.csv',header=None) #1802 is acc_xxxxx ie xxxxx of last csv file for Bearing1_3
print('Preprocessing data for Bearing2_4')
X_test_24 = pd.read_csv('bearing2_4.csv',header=None) #1802 is acc_xxxxx ie xxxxx of last csv file for Bearing1_3
print('Preprocessing data for Bearing2_5')
X_test_25 = pd.read_csv('bearing2_5.csv',header=None) #1802 is acc_xxxxx ie xxxxx of last csv file for Bearing1_3
print('Preprocessing data for Bearing2_6')
X_test_26 = pd.read_csv('bearing2_6.csv',header=None) #1802 is acc_xxxxx ie xxxxx of last csv file for Bearing1_3
print('Preprocessing data for Bearing2_7')
X_test_27 = pd.read_csv('bearing2_7.csv',header=None) #1802 is acc_xxxxx ie xxxxx of last csv file for Bearing1_3
print('Preprocessing data for Bearing3_3')
X_test_33 = pd.read_csv('bearing3_3.csv',header=None) #1802 is acc_xxxxx ie xxxxx of last csv file for Bearing1_3

from sklearn.gaussian_process import GaussianProcessRegressor
print("Training Model_1:\n")
model_1 = GaussianProcessRegressor().fit(X_train_1, t1)#GP model for condition 1
print("Training Model_2:\n")
model_2 = GaussianProcessRegressor().fit(X_train_2, t2)#GP model for condition 2
print("Training Model_3:\n")
model_3 = GaussianProcessRegressor().fit(X_train_3, t3)#GP model for condition 3

output13 = 1-model_1.predict(X_test_13) #change model according to condition
output14 = 1-model_1.predict(X_test_14) #change model according to condition
output15 = 1-model_1.predict(X_test_15) #change model according to condition
output16 = 1-model_1.predict(X_test_16) #change model according to condition
output17 = 1-model_1.predict(X_test_17) #change model according to condition
output23 = 1-model_2.predict(X_test_23) #change model according to condition
output24 = 1-model_2.predict(X_test_24) #change model according to condition
output25 = 1-model_2.predict(X_test_25) #change model according to condition
output26 = 1-model_2.predict(X_test_26) #change model according to condition
output27 = 1-model_2.predict(X_test_27) #change model according to condition
output33 = 1-model_3.predict(X_test_33) #change model according to condition

output13[output13<0] = 1 #remove -ve values
output14[output14<0] = 1 #remove -ve values
output15[output15<0] = 1 #remove -ve values
output16[output16<0] = 1 #remove -ve values
output17[output17<0] = 1 #remove -ve values
output23[output23<0] = 1 #remove -ve values
output24[output24<0] = 1 #remove -ve values
output25[output25<0] = 1 #remove -ve values
output26[output26<0] = 1 #remove -ve values
output27[output27<0] = 1 #remove -ve values
output33[output33<0] = 1 #remove -ve values

ans13 = output13[np.argmin(output13)] * 100 # % RUL
ans14 = output14[np.argmin(output14)] * 100 # % RUL
ans15 = output15[np.argmin(output15)] * 100 # % RUL
ans16 = output16[np.argmin(output16)] * 100 # % RUL
ans17 = output17[np.argmin(output17)] * 100 # % RUL
ans23 = output23[np.argmin(output23)] * 100 # % RUL
ans24 = output24[np.argmin(output24)] * 100 # % RUL
ans25 = output25[np.argmin(output25)] * 100 # % RUL
ans26 = output26[np.argmin(output26)] * 100 # % RUL
ans27 = output27[np.argmin(output27)] * 100 # % RUL
ans33 = output33[np.argmin(output33)] * 100 # % RUL
ans = [ans13,ans14,ans15,ans16,ans17,ans23,ans24,ans25,ans26,ans27,ans33]

name = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7','Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7','Bearing3_3',]

for i in range(len(ans)):
    print('RUL % for',name[i],':',ans[i])