# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 20:17:58 2021

@author: Kalte-PC
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 22:52:07 2021

@author: Kalte-PC
"""
#from IPython import get_ipython
#get_ipython().magic('reset -sf')
import numpy as np
import numpy.matlib
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyDOE import lhs
import time
from copy import deepcopy
import subprocess
import os
use_gpu = True
#torch.cuda.empty_cache()
torch.set_default_tensor_type(torch.DoubleTensor)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

class PINN(nn.Module):
    def __init__(self, x, t, u_train, x_f, t_f, layer_size, neur_size, in_size, out_size, params_list = None):
        
        super(PINN, self).__init__()
        
        self.x = x
        self.t = t
        self.u_train = u_train
        self.x_f = x_f
        self.t_f = t_f
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(in_size, neur_size))
        
        for k in range(layer_size):
            self.layers.append(nn.Linear(neur_size, neur_size))
        self.out = nn.Linear(neur_size, out_size)
        #self.layers.append(nn.Linear(neur_size, out_size))
        
        count = 0
        for m in self.layers: #define the weight and bias in each layer
            if params_list is None:
                # nn.init.normal_(m.weight, mean = 0., std = 1/np.sqrt(layer_size))
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)
            else:
                m.weight = params_list[count]   #if a params_list is given and the order is weight, bias, weight, bias
                m.bias = params_list[count + 1]
                count += 1 
            
        if params_list is None:
            # nn.init.normal_(self.out.weight, mean = 0., std = 1/np.sqrt(layer_size))
            nn.init.xavier_normal_(self.out.weight, gain=1.0)
            nn.init.constant_(self.out.bias, 0.0)
        else:
            self.out.weight = params_list[-2]
            self.out.bias = params_list[-1]         
         
        
    def fpass(self, x):#a forward pass for Coordinates (x, t)
        for m_layer in self.layers:
            x = torch.tanh(m_layer(x)).cuda()   
        output = self.out(x)
        return output

    def NN_U(self, x, t):
        x = torch.from_numpy(x).cuda()
        t = torch.from_numpy(t).cuda()
        X = torch.stack((x, t), 1) #arrange coordinates in column direction
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = self.fpass(X).cuda()
        #print(u.size())
        return u
    
    def NN_F_auto(self, x, t):
        x = torch.from_numpy(x).cuda()
        t = torch.from_numpy(t).cuda()
        x.requires_grad_(True)
        t.requires_grad_(True)
        X = torch.stack((x, t), 1) #arrange coordinates in column direction
        #print(x.size())
        u = self.fpass(X).cuda()
        wei = torch.ones(u.size()).cuda()
        u.requires_grad_(True)
        u_t = torch.zeros(u.size())
        u_t = torch.autograd.grad(outputs=u,
                                  inputs=t,
                                  grad_outputs=wei,
                                  retain_graph=True,
                                  create_graph=True)#,
                                  #only_inputs=True)
        u_x = torch.zeros(u.size()).cuda()
        u_x = torch.autograd.grad(outputs=u,
                                  inputs=x,
                                  grad_outputs=wei,
                                  retain_graph=True,
                                  create_graph=True)#,
                                  #only_inputs=True)
        u_xx = torch.zeros(u.size()).cuda()
        wei2 = torch.ones(u[:,0].size()).cuda()
        u_xx = torch.autograd.grad(outputs=u_x,
                                   inputs=x,
                                   grad_outputs=wei2,
                                   retain_graph=True,
                                   create_graph=True)#,
                                   #only_inputs=True)

        nu = 0.01/np.pi
        u_t2 = torch.stack(u_t,1)
        u_x2 = torch.stack(u_x,1)
        u_xx2 = torch.stack(u_xx,1)
        f = u_t2 + u*u_x2 - nu*u_xx2
        #print(torch.is_tensor(x))
        #f2 = u*u_x2
        #print(f2.size())
        #print(f.size(),u.size(),u_t2.size(),u_x2.size(),u_xx2.size())
        #print(u_x)
        #print(f)
        #u_t = torch.zeros(u_t2.size())
        #print(u)
        #u_x = torch.zeros(u_x2.size())
        #u_xx = torch.zeros(u_xx2.size())
        return f

    
    def NN_F2(self, x, t):
        x = torch.from_numpy(x).cuda()
        t = torch.from_numpy(t).cuda()
        x.requires_grad_(True)
        t.requires_grad_(True)        
        X = torch.stack((x, t), 1) #arrange coordinates in column direction

        u = self.fpass(X).cuda()
        
        u_t = torch.sum(self.gradcal(u, t), dim =1)
        u_x = torch.sum(self.gradcal(u, x), dim =1)
        u_x.requires_grad_(True)
        u_xx = torch.sum(self.gradcal(u_x, x), dim =1)
        
        nu = 0.01/np.pi

        f = u_t + u*u_x - nu*u_xx
        return f
    
    def gradcal(self, f, x):
        assert x.requires_grad
        
        f_shape = f.shape
        x_shape = x.shape
        
        f = f.view(-1)
        x_grads = []
        #global con
        #con = 0
        for f_val in f:
            if x.grad is not None:
                x.grad.data.zero_()
            f_val.backward(retain_graph = True)
            if x.grad is not None:
                x_grads.append(deepcopy(x.grad.data))
            else:
                x_grads.append(torch.zeros(x.shape).to(x))
            output_shape = list(f_shape) + list(x_shape)
            print('count', con)
            #con = con + 1
        return torch.cat((x_grads)).view(output_shape)
                         
    def error(self, u, u_train, f):
        u_train = torch.from_numpy(u_train).cuda()
        error0 = torch.mean(torch.square(u-u_train))
        errorf = torch.mean(torch.square(f))
        error = 1.0*error0 + 1.0*errorf
        return error
    
    def train(self, epochs):
        ax = []
        Error = []        
        plt.ion()
        for epoch in range(epochs):            
            global con
            con = 0
            def closure():
                u = self.NN_U(self.x, self.t).cuda()
                f = self.NN_F_auto(self.x_f, self.t_f).cuda()
                # f = self.NN_F2(self.x_f, self.t_f).cuda()
                error_o = self.error(u, self.u_train, f).cuda()
                global temp 
                temp = error_o
                global con
                print('Count,', con, 'CloasureError', error_o)
                ax.append(con)
                Error.append(temp)
                plt.plot(ax, Error)
                optimizer.zero_grad()
                error_o.backward()
                con = con + 1
                return error_o
            optimizer.step(closure)
            #get_ipython().magic('remove all plots -sf')
            #ax.append(epoch)
            print('\n', 'Epoch', epoch, 'Error', temp, '\n')
            #Error.append(temp)
            if epoch >= 0:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': Error[epoch],
                            }, r'.\ModelLBFGS.tar')                 
                # if Error[epoch] < Error[epoch-1]:           
                #     global Nr
                #     Nr = epoch
                #     print('Epoch',Nr)
               
            plt.clf()# 清除之前画的图
            plt.plot(ax, Error)# 画出当前 ax 列表和 ay 列表中的值的图形
            plt.pause(0.5)
            plt.clf()
            plt.ioff()
        plt.plot(ax, Error)
        plt.savefig('ErrorLBFGS.png', dpi = 150)
        np.savetxt(r'.\ErrorLBFGS.dat', Error)

    def train1(self, epochs):
        ax = []
        Error = []        
        plt.ion()
        for epoch in range(epochs):            
            u = self.NN_U(self.x, self.t).cuda()
            f = self.NN_F_auto(self.x_f, self.t_f).cuda()
            #f = self.NN_F2(self.x_f, self.t_f).cuda()
            error_o = self.error(u, self.u_train, f).cuda()
            optimizer1.zero_grad()
            error_o.backward()
            optimizer1.step()
            #get_ipython().magic('remove all plots -sf')
            ax.append(epoch)
            print('Epoch', epoch, 'Error', error_o)
            Error.append(error_o)
            if epoch > 0:
                if Error[epoch] < Error[epoch-1]:             
                    global Nr                    
                    Nr = epoch
                    print('Epoch',Nr)
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer1.state_dict(),
                                'loss': Error[epoch],
                                }, r'.\Model2.tar')         
            plt.clf()# 清除之前画的图
            plt.plot(ax, Error)# 画出当前 ax 列表和 ay 列表中的值的图形
            plt.pause(0.5)
            plt.clf()
            plt.ioff()
        plt.plot(ax, Error)
        plt.savefig('Error1.png', dpi = 150)
        np.savetxt(r'.\Error1.dat', Error)
        
#training
noise = 0.0

N_u = 100 #training data amount
N_f = 10000 # collocation points amount
layers = 8#[2, 20, 20, 20, 20, 20, 20, 1] #9 in total, assigning only the first 8 layers

data = scipy.io.loadmat('burgers_shock.mat')

t = data['t'].flatten()[:,None] #t is 100x1, after flatten become a matrix of 1x100? Seems not
x = data['x'].flatten()[:,None]
Sol = np.real(data['usol']).T

X, T = np.meshgrid(x,t) #map the x, t into a 2D grid, with coordinates (x,t), X, T are the coordinates matrix
X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Sol.flatten()[:,None]
#codes above arranges u_star and X_star(coordinates) accordingly.

Lb = X_star.min(0) #the boundary x=-1
Ub = X_star.max(0) #the boundary x=1

leftb = np.hstack((X[0:1,:].T, T[0:1,:].T)) # t=0, 0.01
u_leftb = Sol[0:1,:].T

lowerb = np.hstack((X[:,0:1], T[:,0:1])) #x=-1, -0.9x
u_lowerb = Sol[:,0:1] 
u_lowerb2 = Sol[:,0:2]
upperb = np.hstack((X[:,-1:], T[:,-1:])) #[-1:] return the last column or row, x = 0.9x, 1
u_upperb = Sol[:,-1:]

X_u_train = np.vstack([leftb, lowerb, upperb])
X_f_train = Lb + (Ub-Lb)*lhs(2, N_f) #sampling on the left boundary(between x=-1 and x=1) for N_f positions
X_f_train = np.vstack((X_f_train, X_u_train))
u_train = np.vstack([u_leftb, u_lowerb, u_upperb])
    
idx = np.random.choice(X_u_train.shape[0], N_u, replace = False)# boundary sampling 
X_u_train2 = X_u_train[idx, :]#boundary samples
u_train2 = u_train[idx, :]

# model0 = PINN(X_u_train2[:,0].T, 
#               X_u_train2[:,1].T, 
#               u_train2, 
#               X_f_train[:,0].T,
#               X_f_train[:,1].T,
#               layers, 
#               20, 
#               2, 
#               1).cuda
model = PINN(X_u_train2[:,0], 
             X_u_train2[:,1], 
             u_train2, 
             X_f_train[:,0],
             X_f_train[:,1],
             layers, 
             20, 
             2, 
             1).cuda()
optimizer = optim.LBFGS(params = model.parameters(), lr = 1, max_iter = 2000, tolerance_grad = 1e-07, tolerance_change = 1e-09, history_size = 10, line_search_fn=None)#max_eval = None)
optimizer1 = optim.SGD(params = model.parameters(), lr = 0.001, weight_decay= 0.0)#max_eval = None)

epoch = 1
Switch = 0 # 1 Closure LBFGS, 0 Non closure SGD Adam etc

if Switch == 0:
    model.train(epoch)
if Switch == 1:
    model.train1(epoch*2)

#test data
dx0 = np.arange(-1, 1., 0.01)
dt0 = np.arange(0, 1.0, 0.001)
X1, T1 = np.meshgrid(dx0,dt0)
Xx = np.hstack(X1)
Tt = np.hstack(T1)
XxT = torch.from_numpy(Xx).cuda()
TtT = torch.from_numpy(Tt).cuda()

X2 = (torch.stack((XxT, TtT), 1))
#fig, ax = newfig(1.0, 1.1)
with torch.no_grad():
    U_final = model.NN_U(Xx, Tt)
U_final = U_final.cpu()
U_final2 = U_final.detach().numpy()
Results = np.vstack((Tt, Xx, U_final2[:,0])).T
np.savetxt(r'.\U_final2.dat', Results)
#plt.show()
#os.system('python plot.py')
#U_final3 = griddata(X2.detach().numpy(), U_final2, (X1, T1), method = 'nearest')
#h = plt.imshow(U_final3.T, interpolate = 'nearest', cmap = 'rainbow', extent = [0, 0.99, -1, 1], origin = 'lower', aspect = 'auto')

plt.figure() 
plt.scatter(Tt, Xx, c = U_final2, cmap='rainbow')
plt.scatter(X_u_train2[:,1].T, X_u_train2[:,0].T, c = u_train2, cmap='rainbow')
plt.savefig('U_final2.png', dpi = 150)
plt.show()
# plt.scatter(Tt, Xx, c = U_final2, cmap='coolwarm')
# plt.legend()
# plt.show()
# Map = np.zeros((len(Xx), 3))
# Map[:,0] = Tt
# Map[:,1] =Xx
# Map[:,1] =U_final2[:,0]
# Mat = np.mat(Map)
# #[Tt, Xx, U_final2]