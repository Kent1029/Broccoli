import numpy as np
from numpy import linalg as LA
import os,sys
import argparse


import re

parser = argparse.ArgumentParser(description='clustering agreement measures')

parser.add_argument('--data_groundtruth',  help='data file directory')
parser.add_argument('--data_results',  help='data file directory')




args = parser.parse_args()



# open your csv and read as a text string
with open(args.data_results, 'r') as f:
    my_csv_text = f.read()

find_str = 'ï»¿1'
replace_str = ''

# substitute
new_csv_str = re.sub(find_str, replace_str, my_csv_text)

# open new file and save
new_csv_path = './my_new_csv.csv' # or whatever path and name you want
with open(new_csv_path, 'w') as f:
    f.write(new_csv_str)






Y = np.loadtxt(open(args.data_results, "rb"), delimiter=",")
#Y_true = np.loadtxt(open('./my_new_csv.csv', "rb"), delimiter=",")
Y_true = str()
with open(args.data_groundtruth, "rb") as f:
  lines = f.readlines()
  lines[0] = lines[0][3:]
  Y_true = np.loadtxt(lines, delimiter=",")


Ym,Yn = Y.shape


















'''
cuda = torch.cuda.is_available()
dev = torch.device("cuda") if cuda else torch.device("cpu")

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--data',  help='data file directory')
parser.add_argument('--X',  help='directory to ground truth X')
parser.add_argument('--C',  help='directory to ground truth C')
parser.add_argument('--Y',  help='directory to ground truth Y')
parser.add_argument('--r', default=3, type=int, help='rank - number of clusters')
parser.add_argument('--run', default='_', help='run specification, the results are saved in the path results_run/...')
args = parser.parse_args()

D = np.loadtxt(open(args.data, "rb"), delimiter=",")
Y = np.loadtxt(open(args.Y, "rb"), delimiter=",")
C = np.loadtxt(open(args.C, "rb"), delimiter=",")
X = np.loadtxt(open(args.X, "rb"), delimiter=",")
m,n = D.shape
r = args.r
'''


def Frobenius_norm(A):
    return (np.trace(np.transpose(A).dot(A)))**0.5


def Frobenius_norm2(A):
    return LA.norm(A)

"""
def F1(A,A2):
    temp = 0
    for i in range(A.shape[0]):
        temp+= (2 * np.transpose(A[i,:]).dot(A2[i,:]))/(A[i,:].dot(np.transpose(A[i,:]))+A2[i,:].dot(np.transpose(A2[i,:])))
    
    return temp/A.shape[0]
"""
def F1(A,A2):
    temp = 0
    for i in range(A.shape[1]):
        temp+= (2 * np.transpose(A[:,i]).dot(A2[:,i]))/(A[:,i].dot(np.transpose(A[:,i]))+A2[:,i].dot(np.transpose(A2[:,i])))
    return temp/A.shape[1]


def I_cos(A,A2):
    return Frobenius_norm(np.transpose(A).dot(A2))**2/(Frobenius_norm(A.dot(np.transpose(A)))*Frobenius_norm(A2.dot(np.transpose(A2))))


def I_cos2(A,A2):
    return Frobenius_norm2(np.transpose(A).dot(A2))**2/(Frobenius_norm2(A.dot(np.transpose(A)))*Frobenius_norm2(A2.dot(np.transpose(A2))))


def I_sub(A,A2):
    return Frobenius_norm(np.transpose(A).dot(A2))/(Frobenius_norm(A)*Frobenius_norm(A2))


def I_sub2(A,A2):
    return Frobenius_norm2(np.transpose(A).dot(A2))/(Frobenius_norm2(A)*Frobenius_norm2(A2))



"""
average of the measures:
"""
def avg_F1(Y,Y_true,X,X_true):
    return 0.5*(F1(Y,Y_true)+F1(X,X_true))


def avg_Icos(Y,Y_true,X,X_true):
    return 0.5*(I_cos(Y,Y_true)+I_cos(X,X_true))


def avg_Isub(Y,Y_true,X,X_true):
    return 0.5*(I_sub(Y,Y_true)+I_sub(X,X_true))


#print(Y_true)

print("F1 :",F1(Y,Y_true))
print("I_sub  :",I_sub(Y,Y_true))
print("I_sub2 :",I_sub2(Y,Y_true))
print("I_cos  :",I_cos(Y,Y_true))
print("I_cos2 :",I_cos2(Y,Y_true))
