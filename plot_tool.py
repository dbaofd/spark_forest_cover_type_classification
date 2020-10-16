import matplotlib.pyplot as plt
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
import numpy as np

def plot_bar(bar_width, bars, data_list, title, figure_size):
    fig_1 = plt.figure(figsize=(figure_size[0],figure_size[1]))
    ax = fig_1.add_axes([0,0,1,1])
    x=np.arange(len(bars))+1
    plt.bar(bars,data_list,alpha=0.8,width=bar_width)
    plt.xticks(rotation=90)
    for i in range(len(x)):
        plt.text(x = x[i]-1-(bar_width/2) , y = data_list[i]+0.5,va= 'bottom', s = data_list[i], size = 11)
    plt.title(title)
    plt.show()
    
def plot_pca(myrdd, title, color):
    mat=RowMatrix(myrdd)
    pc = mat.computePrincipalComponents(2)
    # Project the rows to the linear space spanned by the top 2 principal components.
    projected = mat.multiply(pc)
    a=projected.rows.collect()
    sum_pca1=0;
    sum_pca2=0;
    for i in a:
        sum_pca1=sum_pca1+i[0];
        sum_pca2=sum_pca2+i[1];
        plt.plot(i[0],i[1], 'o', color=color)
    ave_pca1=sum_pca1/len(a)
    ave_pca2=sum_pca2/len(a)
    plt.plot(ave_pca1,ave_pca2,'^', markersize=10, color='red')
    plt.title(title)
    plt.show()