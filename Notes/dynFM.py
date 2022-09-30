import networkx as nx
import numpy as np
import math
import random
import os
import sys
import gurobipy as gp
from gurobipy import GRB

#Euclidian Distance between two d-dimensional points
def eucl_dist_sq(p0,p1):
    #print(p0, p1)
    ans = sum([((p0[i]-p1[i]) ** 2) for i in range(len(p0))])
    #print(p0, p1, ans)
    return ans


# return the shortest path distance of each node for a given graph G and a source node
def get_distance(G, source):
	V = len(G.nodes())
	dijkstra_distance = nx.single_source_dijkstra_path_length(G, source)
	max_distance = np.max(list(dijkstra_distance.values()))
	distance = np.ones(V) * (2 * max_distance)
	for node, dist in dijkstra_distance.items():
		distance[node] = dist
	return distance


# embedding a given graph into K-dimensional points
# A: numpy array, an adjecent matrix (V, V) represents an undirected graph: A=A^T
#	A(i, j) is the weight of the edge between node i and node j
#	A(i, j)=0 means there is no edge between node i and node j
# K: user-specifeid K-dimensional space for the embedding
# e: a threshold for early stop (the default value can be used directly)
# q: the power of q
# return: numpy array with size (V, K)
def fastmap_L2(G, K, e=0.0001, q=0.5):
	K_hat = K
	C = 10
	V = len(G.nodes())
	p = np.zeros((V, K))
	for r in range(K):
		v_a = random.randrange(V)
		v_b = v_a
		for t in range(C):
			d_ai = np.power(get_distance(G, v_a), q)
			#print(r, v_a, d_ai)
			distance = np.zeros(V)
			for v_i in range(V):
				distance[v_i] = (d_ai[v_i])**2 - np.sum(np.power(p[v_a, :r] - p[v_i, :r], 2))
			v_c = np.argmax(distance)
			if v_c == v_b:
				break
			else:
				v_b = v_a
				v_a = v_c
		#print(r, v_a, v_b)
		d_ai = np.power(get_distance(G, v_a), q)
		d_ib = np.power(get_distance(G, v_b), q)
		d_ab_new = (d_ai[v_b])**2 - np.sum(np.power(p[v_a, :r] - p[v_b, :r], 2))
		if d_ab_new < e:
			K_hat = r
			break
		for v_i in range(V):
			d_ai_new = (d_ai[v_i])**2 - np.sum(np.power(p[v_a, :r] - p[v_i, :r], 2))
			d_ib_new = (d_ib[v_i])**2 - np.sum(np.power(p[v_i, :r] - p[v_b, :r], 2))
			p[v_i, r] = (d_ai_new + d_ab_new - d_ib_new) / (2 * np.sqrt(d_ab_new))
			"""
			print(r, end =" ")
			print(d_ai_new, end =" ")
			print(d_ib_new, end =" ")
			print(p[v_i, r])
			"""
	return p


# Rotate 'new points' to match the 'old points' as closely as possible:
def compute_rotation_matrix(points_old, points_new):
	n = points_old[:,0].size
	d = points_old[0].size
	#print(points_old)
	#print(points_new)
	try:

		# Create a new model
		env = gp.Env(empty=True)
		env.setParam("OutputFlag",0)
		env.start()
		m = gp.Model("qp")
		m.params.NonConvex = 2
		m.Params.LogToConsole = 0

		# Create variables (x and y are referred to as A and b in the paper)
		x = m.addMVar((d,d), lb=-1, ub=1, vtype=GRB.CONTINUOUS, name="x")
		z = [m.addMVar(d, vtype=GRB.CONTINUOUS, name="z"+str(i)) for i in range(n)] #denotes difference Ax + b - y
		y = m.addMVar(d, vtype=GRB.CONTINUOUS, name="y")

		# Set objective (minimize sum of squared deviations)
		obj = sum(z[i] @ z[i] for i in range(n))
		m.setObjective(obj, gp.GRB.MINIMIZE)

		# Add constraints:
		# 1) assign meaning to z (deviations in points after rotation)
		for i in range(n):
			m.addConstrs(z[i][j] >= x[j] @ points_new[i] + y[j] - points_old[i][j] for j in range(d))
			m.addConstrs(z[i][j] >= points_old[i][j] - x[j] @ points_new[i] - y[j] for j in range(d))
		
		# 2) x should be orthogonal
		for i in range(d):
			m.addConstr(x[:,i] @ x[:,i] == 1)
		for i in range(d):
			for j in range(i+1, d):
				#print("adding constraint for zero dot " + str(i) + str(j))
				m.addConstr(x[:,i] @ x[:,j] == 0)


		#print('Added constraints')
		# Optimize model
		m.write("lpfile.lp")
		m.optimize()
		#print('tried to optimize rotation matrix')
		if (m.status != gp.GRB.OPTIMAL):
			print('compute iis because status is %g' % m.status)
			m.computeIIS()
			m.write("iis.ilp")
			return
		else:
			rotated_points = np.full((n,d), 0.0)
			#uncomment the print statements for debugging if needed:
			for i in range(n):
				#print(z[0].x + points_old[0])
				rotated_points[i] = np.matmul(x.X, points_new[i]) + np.array(y.X)
				#print(points_old[i], rotated_points[i], z[i].X)
			#for v in m.getVars():
				#print('%s = %g' % (v.varName, v.x))
			#print('Obj: %g' % obj.getValue())
			#print(m.objVal)
			return rotated_points

	except gp.GurobiError as e:
		print('Error code ' + str(e.errno) + ': ' + str(e))

	except AttributeError as e:
		print('Encountered an attribute error: ' + str(e))

if __name__ == '__main__':
	K, q, e = 3, 1, 0.0001
	
	c = np.array([[[1,5,0], [4,8,0], [4,2,0], [7,7,0], [7,3,0]],
		[[1,5,0], [3,9,0], [3,1,0], [6,7,0], [6,3,0]]])

	A = np.full((2, 5, 5), np.inf)			#adjecency matrix
	for t in range(2):
		for i in range(5):
			A[t, i, i] = 0
	
	for t in range(2):
		for i in range(1):
			for j in range(1, 5):
				A[t,i,j] = math.sqrt(eucl_dist_sq(c[t][i], c[t][j]))
				A[t,j,i] = A[t,i,j]

	G0 = nx.from_numpy_matrix(A[0])
	P0 = fastmap_L2(G0, K, e, q)
	G1 = nx.from_numpy_matrix(A[1])
	P1 = fastmap_L2(G1, K, e, q)
	compute_rotation_matrix(P0, P1)
