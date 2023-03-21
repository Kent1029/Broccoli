# Broccoli
[實驗環境]

透過Anaconda3建立python虛擬環境。

Python3.9(不能用Python3.5) 

Pytorch 1.10

GPU  NVIDIA-RTX3080

CUDA 11.3

Numpy

建置在Broccoli_env.yaml


[資料集來源]

http://mulan.sourceforge.net/datasets-mlc.html


[運行程式碼]

python main_initNMF.py --data [data file directory] --r [rank - number of clusters] --lam_C_perc [L2 regularization weight for C] --res_path [path to the folder where the result is to be stored]

python main_initRND.py --data [data file directory] --r [rank - number of clusters] --lam_C_perc [L2 regularization weight for C] --res_path [path to the folder where the result is to be stored]      
