import os
import open3d as o3d
import numpy as np
import pymesh
import trimesh as tri
from numpy import linalg as LA
from numpy.linalg import norm
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import trimesh as tri
import sys
import numpy as np 
import random



class Classifier:
    def unison_shuffling(self,a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def getEigenValues(self, points):
        mean = points.mean(axis=0)
        cov = np.matmul(np.transpose(points-mean), (points-mean))
        w,v = LA.eig(cov)
        w = np.sort(w)
        return w

    def BFS(self, s, g, k):
        # Mark all the vertices as not visited
        visited = [False] * len(g.nodes)
        # Create a queue for BFS
        queue = []
        neighs = []
        queue.append((s, 0))
        visited[s] = True
        dist  = 0
        while (dist < k) and queue:
            both = queue.pop(0)
            s = both[0]
            dist = both[1]
            neighs.append(s)
            for node in g.adj[s]:
                if visited[node] == False:
                    queue.append((node, dist+1))
                    visited[node] = True
        return neighs

    def findFeatures1(self,fileName):
        mesh = tri.load(fileName)
        ed = np.array(mesh.edges_unique)
        np.set_printoptions(threshold=sys.maxsize)
        g = nx.from_edgelist(mesh.edges_unique)
        mesho3 = o3d.io.read_triangle_mesh(fileName)
        points = np.asarray(mesho3.vertices)
        pointsRand = [random.randrange(1, len(g.nodes), 1) for i in range(100)]
        allw = []
        for p in pointsRand:
            neighs = self.BFS(p, g, 5)
            neighborPoints = points[neighs]
            w = self.getEigenValues(points = neighborPoints)
            allw.append(w)
        cov = np.matmul(np.transpose(allw), allw)
        covDiag = cov.diagonal() 
        sum = covDiag[0] + covDiag[1] + covDiag[2]
        features = [covDiag[0]/sum, covDiag[1]/sum, covDiag[2]/sum]
        return features

    def findFeatures2(self,fileName):
        mesho3 = o3d.io.read_triangle_mesh(fileName)
        points = np.asarray(mesho3.vertices)
        mean = points.mean(axis=0)
        cov = np.matmul(np.transpose(points-mean), points-mean)
        w,v = LA.eig(cov)
        w = np.sort(w)
        feature = [w[2]/(w[0] + w[1] + w[2])]
        return feature
        
    def __init__(self, filePathTraining):
        self.X = []
        self.Y = []
        for filename in os.listdir(filePathTraining + "leaf/"):
            print("training on LEAF "+filename)
            features = self.findFeatures2(filePathTraining + "leaf/" + filename)
            self.X.append(np.array(features))
            self.Y.append(1)
        for filename in os.listdir(filePathTraining + "stem/"):
            print("training on STEM "+filename)
            features = self.findFeatures2(filePathTraining + "stem/" + filename)
            self.X.append(np.array(features))
            self.Y.append(0)
        self.X = [np.array(self.X[i]) for i in range(len(self.X))]
        Xtemp = np.array(self.X)
        Ytemp = np.array(self.Y)
        self.clf = svm.SVC()
        Xtemp, Ytemp = self.unison_shuffling(Xtemp, Ytemp)
        self.clf.fit(Xtemp, Ytemp)
        
    def classify(self, fileName):
        feat = self.findFeatures2(fileName)
        featnp = np.array(feat)
        featnp = featnp.reshape(1,-1)
        #print(featnp.shape)
        #print(self.clf.predict(featnp))
        ans = self.clf.predict(featnp)
        return ans[0]
        







