import numpy as np
import sys
from scipy.sparse import csgraph
import time
import pandas as pd
import copy
import os

class Graph_Learn:

    # A - R12, B - R23, Aff - Entire Affinity Matrix
    def __init__(self, A, B, norm=True, iter=2, cluster=5, mu=10, rho = 1, delta = 0.1,
                 p=2, lambda0 = 1000,lambda1=1000,lambda2=1000):
        self.norm = norm
        self.iter = iter
        self.cluster = cluster
        self.mu = mu
        self.rho = rho
        self.delta = delta
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.p = p
        self.A = A
        self.B = B
        self.SA = np.block([
            [np.zeros((A.shape[0],A.shape[0])), A],
            [A.T, np.zeros((A.shape[1],A.shape[1]))]
        ])
        self.SB = np.block([
            [np.zeros((B.shape[0],B.shape[0])), B],
            [B.T, np.zeros((B.shape[1],B.shape[1]))]
        ])
        self.Aff = np.block([
            [np.zeros((A.shape[0],A.shape[0])), A, np.zeros((A.shape[0],B.shape[1]))],
            [A.T, np.zeros((B.shape[0],A.shape[1])), B],
            [np.zeros((B.T.shape[0], A.T.shape[1])), B.T, np.zeros((B.T.shape[0],B.shape[1]))]
        ])

    def P_S_update(self,S,Lambda_S):
        Diff_Med = S - self.Aff
        D_list = []
        for i in Diff_Med:
            D_S_i = (self.p/2) * (i.T@i + self.delta)**((self.p-2)/2)
            D_list.append(D_S_i)

        D = np.diag(np.array(D_list))
        P = (1/self.mu) * D @ (self.Aff - S) + S - Lambda_S/self.mu
        S = (1/self.mu) * D @ (self.Aff - P) + P + Lambda_S/self.mu
        return P,S

    def E0_F0_update(self, X, E0, F0, Lambda_0):
        LSp = csgraph.laplacian(X, normed=self.norm) ** self.p
        M0 = F0 - (1/self.mu)*Lambda_0 - (self.lambda0/self.mu)* LSp@F0
        for id in np.ndindex(E0.shape):
            if M0[id] >= 0:
                E0[id] = M0[id]
            else: E0[id] = 0
        M0_T = (1/self.mu)*E0.T + (1/self.mu)*Lambda_0.T - (self.lambda0/self.mu)* E0.T@LSp
        P1, _, P2T = np.linalg.svd(M0_T.T, full_matrices=False)
        F0 = P1 @ P2T
        return E0,F0

    def Lambda_update(self,Lambda_value, S1,S2):
        return Lambda_value+self.mu*(S1-S2)

    def mu_update(self):
        return self.rho*self.mu

    def F_ini(self, X):
        L = csgraph.laplacian(X, normed=self.norm)
        eigenVal, eigenVec = np.linalg.eig(L)
        idx = eigenVal.argsort()[:self.cluster]
        assert self.cluster == len(list(idx))
        eigenVec = eigenVec[:, idx]
        assert eigenVec.shape[0] == X.shape[0]

        return eigenVec

    def Lambda_ini(self,shape):
        zeros = np.zeros(shape)
        random_diag = list(np.random.rand(min(shape)))
        np.fill_diagonal(zeros,random_diag)

        return zeros

    def update(self):
        n1, n2 = self.A.shape
        n3 = self.B.shape[1]
        n = n1+ n2+ n3
        Aff = copy.deepcopy(self.Aff)
        ### Initialization
        S = np.random.rand(n,n)
        F0 = self.F_ini(self.Aff)
        F1 = self.F_ini(self.SA)
        F2 = self.F_ini(self.SB)
        print('F done')
        E0 = np.random.rand(*F0.shape)
        E1 = np.random.rand(*F1.shape)
        E2 = np.random.rand(*F2.shape)
        print('E done')
        Lambda_S = self.Lambda_ini(self.Aff.shape)
        Lambda_F0 = self.Lambda_ini(F0.shape)
        Lambda_F1 = self.Lambda_ini(F1.shape)
        Lambda_F2 = self.Lambda_ini(F2.shape)
        print('Lambda done')
        ### update
        for i in range(self.iter):
            P,S = self.P_S_update(S,Lambda_S)
            E0,F0 = self.E0_F0_update(self.Aff, E0, F0, Lambda_F0)
            E1,F1 = self.E0_F0_update(self.SA,E1,F1,Lambda_F1)
            E2,F2 = self.E0_F0_update(self.SB,E2,F2,Lambda_F2)
            Lambda_S = self.Lambda_update(Lambda_S, P,S)
            Lambda_F0 = self.Lambda_update(Lambda_F0, E0, F0)
            Lambda_F1 = self.Lambda_update(Lambda_F1, E1, F1)
            Lambda_F2 = self.Lambda_update(Lambda_F2, E2, F2)
            self.mu = self.mu_update()
            print('iter', i)
        assert S.shape == self.Aff.shape
        return S,Aff

def save_sub(Aff, timestr, n2, n3, n1=26):
    A12 = Aff[:n1,n1:(n1+n2)]
    A23 = Aff[n1:(n1+n2),(n1+n2):(n1+n2+n3)]
    path_12 = 'Data/Matrices/Matrix_R12.csv'
    A0 = pd.read_csv(path_12)
    path_23 = 'Data/Matrices/Matrix_R23.csv'
    B0 = pd.read_csv(path_23)
    human = list(A0.columns.values)[1:]
    virus = list(A0.iloc[:,0])
    drug = list(B0.columns.values)[1:]
    human_ = human[:n2]
    virus_ = virus[:n1]
    drug_ = drug[:n3]
    os.chdir('PPI/covid_ppi/results')
    hvi = pd.DataFrame(data = A12, index=virus_,columns=human_).to_csv(f'hvi_{timestr}.csv')
    dhi = pd.DataFrame(data = A23, index=human_,columns=drug_).to_csv(f'dhi_{timestr}.csv')

if __name__ == '__main__':
    start = time.time()
    np.set_printoptions(threshold=sys.maxsize)
    timestr = time.strftime("%Y%m%d_%H_%M_%S")
    print(timestr)
    # Set Affinity Size
    n1 = 26
    n2 = int(input("Enter n2: "))
    n3 = int(input("Enter n3: "))
    #Input A (26 * 16872)
    path_12 = 'Data/Matrices/Matrix_R12_0.6.npy'
    A = np.load(path_12)
    print(A.shape)
    A = A[:,:n2]
    #Input B (16872 * 8279)
    path_23 = 'Data/Matrices/Matrix_R23.npy'
    B = np.load(path_23)
    print(B.shape)
    B =B[:n2,:n3]
    #Update Graph
    Graph = Graph_Learn(A,B)
    S,A = Graph.update()
    #Export Results
    shape = f'{A.shape[0]}_{A.shape[1]}'
    os.chdir(r'PPI/covid_ppi/results')
    np.save(f'S_{shape}_{timestr}.npy', S)
    np.savetxt(f'S_{shape}_{timestr}.csv', S, delimiter=",")
    np.save(f'Aff_{shape}_{timestr}.npy', A)
    np.savetxt(f'Aff_{shape}_{timestr}.csv', A, delimiter=",")
    save_sub(Aff=S,timestr=timestr,n2=n2,n3=n3,n1=n1)
    #Miscellaneous
    print(f'Total Time Cost: {time.time()-start}')
