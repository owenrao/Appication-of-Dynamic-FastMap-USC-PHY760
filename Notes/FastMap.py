import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class FastMap:
    
    def __init__(self):
        return

    def find_distance(self,a,b):
        return self.dist_mat["value"][self.dist_mat["index"].index({a,b})]

    def find_furthest_pair(self):
        a,b = self.dist_mat["index"][np.argmax(self.dist_mat["value"])]
        return a,b
        
    def gen_distance_matrix(self,x):
        for i in range(len(self.dist_mat["index"])):
            a,b = self.dist_mat["index"][i]
            try:
                self.dist_mat["value"][i] = np.sqrt(np.square(self.dist_mat["value"][i])-np.square(x[a]-x[b]))
            except Warning:
                print((a,b))
                print(np.square(x[a]-x[b]))
                print(self.dist_mat["value"][i])
    
    def gen_x(self,a,b):
        x = np.zeros(self.N)
        dab = self.find_distance(a,b)
        x[a] = 0 # P_a to itself
        x[b] = dab # Furthest distance
        for i in range(self.N):
            if i==a or i==b:
                continue
            dai = self.find_distance(a,i)
            dib = self.find_distance(i,b)
            x[i] = (dai**2+dab**2-dib**2)/(2*dab)
        return x

    def fit(self,dist_mat,keywords_list,k=2):
        self.dist_mat = dist_mat
        self.keywords_list = keywords_list
        self.k = k
        self.N = len(keywords_list)
        result = np.zeros((self.N,k))
        for c in range(k):
            if c != 0:
                dist_mat = self.gen_distance_matrix(x)
            a,b = self.find_furthest_pair()
            x = self.gen_x(a,b)
            result[:,c] = x
        return result

    def plot(map_result,keywords_list):
        #map_result = self.fit(dist_mat,keywords_list,k)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(map_result[:,0], map_result[:,1])
        for (label,x,y) in zip(keywords_list,map_result[:,0],map_result[:,1]):
            plt.annotate(label, (x, y))
        plt.show