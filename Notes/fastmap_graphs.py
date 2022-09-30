import networkx as nx
import numpy as np
import random
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score as get_NMI
import random
import math
from scipy.io import loadmat
import glob
from collections import defaultdict
from scipy.stats import entropy
from collections import Counter
from scipy.io import savemat
import time
import community
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime


def graph_subgraph(adj_matrix, set_num, isBM=True):
	edges_mask = []
	nonedges_mask = []
	V = adj_matrix.shape[0]
	for vi in range(V):
		for vj in range(vi + 1, V):
			nonedges_mask.append((vi, vj)) if adj_matrix[vi, vj] == 0 else edges_mask.append((vi, vj))

	graphs, subgraphs = [], []
	G = nx.Graph()
	G.add_nodes_from(range(V))
	G.add_edges_from(edges_mask)
	graphs.append(G)
	for set_i in range(set_num):
		generate_subgraphs(V, edges_mask, 10, subgraphs, G)
	if isBM:
		if len(edges_mask) >= len(nonedges_mask):
			cG = nx.Graph()
			cG.add_nodes_from(range(V))
			cG.add_edges_from(nonedges_mask)
			graphs.append(cG)
			for set_i in range(set_num):
				generate_subgraphs(V, nonedges_mask, 10, subgraphs, cG)
		else:
			for set_i in range(set_num):
				random.shuffle(nonedges_mask)
				cur_nonedges_mask = nonedges_mask[:len(edges_mask)]
				cG = nx.Graph()
				cG.add_nodes_from(range(V))
				cG.add_edges_from(cur_nonedges_mask)
				subgraphs.append(cG)
				generate_subgraphs(V, cur_nonedges_mask, 10, subgraphs, cG)
	return graphs, subgraphs


def generate_subgraphs(V, edges_mask, N_step, subgraphs, G):
	random.shuffle(edges_mask)
	edges_num = len(edges_mask)
	edges_stepsize = math.ceil(edges_num / N_step)
	pre_G = G
	for n in range(N_step):
		cur_G = pre_G.copy()
		cur_G.remove_edges_from(edges_mask[n * edges_stepsize: min((n + 1) * edges_stepsize, edges_num)])
		subgraphs.append(cur_G)
		pre_G = cur_G


def efficent_distances(graphs, subgraphs, set_num, source):
	distances = get_distances(graphs, source) * set_num
	distances += get_distances(subgraphs, source)
	return distances


def get_distances(graphs, source):
	V = len(graphs[0].nodes())
	dijkstra_distances = []
	distances = np.zeros(V)
	for G in graphs:
		dijkstra_distances.append(nx.single_source_dijkstra_path_length(G, source))
	
	max_distances = []
	for dijkstra_distance in dijkstra_distances:
		max_distances.append(2 * (np.max(list(dijkstra_distance.values()))))

	for v in range(V):
		for dijkstra_distance, max_dis in zip(dijkstra_distances, max_distances):
			if v in dijkstra_distance:
				distances[v] += dijkstra_distance[v]
			else:
				distances[v] += max_dis #V #2*15 - 1
	return distances


#adj_matrix: an adjacent matrix that represents a graph
def fastmap_L2(adj_matrix, K, e, set_num=1, isBM=True):
	K_hat = K
	C = 10
	Gs, subGs = graph_subgraph(adj_matrix, set_num, isBM)
	V = len(Gs[0].nodes())
	p = np.zeros((V, K))
	for r in range(K):
		v_a = random.randrange(V)
		v_b = v_a
		for t in range(C):
			d_ai = efficent_distances(Gs, subGs, set_num, v_a)
			distance = np.zeros(V)
			for v_i in range(V):
				distance[v_i] = (d_ai[v_i])**2 - np.sum(np.power(p[v_a, :r] - p[v_i, :r], 2))
			v_c = np.argmax(distance)
			if v_c == v_b:
				break
			else:
				v_b = v_a
				v_a = v_c
		d_ai = efficent_distances(Gs, subGs, set_num, v_a)
		d_ib = efficent_distances(Gs, subGs, set_num, v_b)
		d_ab_new = (d_ai[v_b])**2 - np.sum(np.power(p[v_a, :r] - p[v_b, :r], 2))
		if d_ab_new < e:
			K_hat = r
			break
		for v_i in range(V):
			d_ai_new = (d_ai[v_i])**2 - np.sum(np.power(p[v_a, :r] - p[v_i, :r], 2))
			d_ib_new = (d_ib[v_i])**2 - np.sum(np.power(p[v_i, :r] - p[v_b, :r], 2))
			p[v_i, r] = (d_ai_new + d_ab_new - d_ib_new) / (2 * np.sqrt(d_ab_new))
	return p[:, :K_hat], Gs


def load_airdata(filename):
	airdata = np.loadtxt(filename, delimiter=' ')
	airnum = int(airdata[0, 0])
	edges = airdata[1:, :2].astype(int) - 1
	adj_matrix = np.zeros((airnum, airnum))
	for edge in edges:
		adj_matrix[edge[0], edge[1]] = adj_matrix[edge[1], edge[0]] = 1
	return adj_matrix


def load_mat(filename):
	mat = loadmat(filename)
	keys = list(mat.keys())
	adj_matrix = np.array(np.absolute(mat[keys[3]]))
	solutions = np.array(mat[keys[4]][0])
	clusters = np.max(solutions) + 1
	return adj_matrix, clusters, solutions


def get_Mmat(adj_matrix, labels, n_parts):
	labels = np.array(labels)
	Mmat = np.zeros((n_parts, n_parts))
	partn_nodes = []
	for part in range(n_parts):
		partn_nodes.append(np.where(labels == part)[0])
	for part1 in range(n_parts):
		part1_nodes = partn_nodes[part1]
		for part2 in range(part1, n_parts):
			part2_nodes = partn_nodes[part2]
			denominator = (len(part1_nodes) * len(part2_nodes)
				if part1 != part2 else len(part1_nodes) * (len(part1_nodes) - 1))
			denominator = np.max((denominator, 1))
			numerator = 0
			for node1 in part1_nodes:
				for node2 in part2_nodes:
					numerator += adj_matrix[node1, node2] if node1 != node2 else 0
			Mmat[part1, part2] = Mmat[part2, part1] = numerator / denominator
	return Mmat


def sample_objective(final_mat, mask):
	score = 0
	for coordinate in mask:
		score += final_mat[coordinate]**2
	return score


def get_score(A, C, M, R):
	final_mat = (A - np.matmul(np.matmul(C, M), C.T)) * (A - R)
	score = np.power(np.linalg.norm(final_mat), 2)
	return score


def single_score(adj_matrix, labels):
	n_parts = np.max(labels) + 1
	Rmat = np.ones(adj_matrix.shape) * (np.sum(adj_matrix) / adj_matrix.size)
	Mmat = get_Mmat(adj_matrix, labels, n_parts)
	Cmat = np.zeros((len(labels), n_parts))
	for node in range(len(labels)):
		Cmat[node, labels[node]] = 1
	final_mat = (adj_matrix - np.matmul(np.matmul(Cmat, Mmat), Cmat.T)) * (adj_matrix - Rmat)
	return np.power(np.linalg.norm(final_mat), 2)


def hard_Cmat(Cmat, labels):
	for i in range(len(labels)):
		Cmat[i, labels[i]] = 1
	return Cmat


def bestscore_choice(adj_matrix, K, e, n_parts, times, graphsets_num, isBM=True):
	start_time = time.time()
	best_score, best_labels, best_embedding = float('inf'), None, None
	Rmat = np.ones(adj_matrix.shape) * (np.sum(adj_matrix) / adj_matrix.size)
	t = 0
	while t < times:
		points, graphs = fastmap_L2(adj_matrix, K, e, graphsets_num, isBM)
		if points.shape[1] != K:
			continue
		GMM = GaussianMixture(n_components=n_parts)
		labels = GMM.fit_predict(points)
		#Cmat = GMM.predict_proba(points)
		Cmat = np.zeros((adj_matrix.shape[0], n_parts))
		Cmat = hard_Cmat(Cmat, labels)
		Mmat = get_Mmat(adj_matrix, labels, n_parts)
		score = get_score(adj_matrix, Cmat, Mmat, Rmat)
		if score < best_score:
			best_score = score
			best_labels = labels
			best_embedding = points
		t += 1
	return best_labels, best_score, time.time() - start_time, graphs[0], best_embedding


def blockmodeling_M(k_parts, V):
	p = np.log(V) / V
	M = np.zeros((k_parts, k_parts))
	count = np.zeros(k_parts)
	strong_connected = np.min((3, k_parts))
	connected_pairs = []
	sparse_pairs = []
	for b1 in range(k_parts):
		for b2 in range(b1, k_parts):
			connected_pairs.append((b1, b2))
	while np.sum(count) < k_parts * strong_connected:
		pair = random.choice(connected_pairs)
		if count[pair[0]] < strong_connected and count[pair[1]] < strong_connected:
			count[pair[0]] += 1
			count[pair[1]] += 1
			M[pair] = M[pair[1], pair[0]] = 10 * p #random.uniform(0.8, 1.0)
		else:
			sparse_pairs.append(pair)
		connected_pairs.remove(pair)
	for pair in sparse_pairs + connected_pairs:
		M[pair] = M[pair[1], pair[0]] = p #random.uniform(0, 0.1)
	return M


def ariportshape_M(k_parts, V):
	p = 1 / V
	#p = np.log(V) / V
	M = np.zeros((k_parts, k_parts))
	count = np.zeros(k_parts)
	strong_connected = np.min((2, k_parts))
	connected_pairs = []
	sparse_pairs = []
	for b1 in range(k_parts):
		for b2 in range(b1 + 1, k_parts):
			connected_pairs.append((b1, b2))
	while (len(connected_pairs) > 0) and (np.sum(count) < k_parts * strong_connected):
		pair = random.choice(connected_pairs)
		if count[pair[0]] < strong_connected and count[pair[1]] < strong_connected:
			count[pair[0]] += 1
			count[pair[1]] += 1
			M[pair] = M[pair[1], pair[0]] = 10 * p
		else:
			sparse_pairs.append(pair)
		connected_pairs.remove(pair)
	for pair in sparse_pairs + connected_pairs + [(b, b) for b in range(k_parts)]:
		M[pair] = M[pair[1], pair[0]] = p
	return M


def communitydetection_M(k_parts, V):
	p = np.log(V) / V
	M = np.zeros((k_parts, k_parts))
	for b1 in range(k_parts):
		for b2 in range(b1, k_parts):
			M[b1, b2] = M[b2, b1] = 10 * p if b1 == b2 else p
	return M


def artificial_MC(n_nodes, k_parts, isBM=True, isAP=False):
	tempC = np.random.rand(n_nodes, k_parts)
	membership = np.argmax(tempC, axis=1)
	C = np.zeros((n_nodes, k_parts))
	for node in range(n_nodes):
		C[node, membership[node]] = 1
	if not isAP:
		M = blockmodeling_M(k_parts, n_nodes) if isBM else communitydetection_M(k_parts, n_nodes)
	else:
		M = ariportshape_M(k_parts, n_nodes)
	return M, C


def artificial_A(M, C):
	labels = np.where(C == 1)[1]
	n_nodes, k_parts = C.shape[0], C.shape[1]
	A = np.zeros((n_nodes, n_nodes))
	for v1 in range(n_nodes):
		for v2 in range(v1 + 1, n_nodes):
			A[v1, v2] = A[v2, v1] = 1 if M[labels[v1], labels[v2]] >= np.random.random() else 0
	return A


def add_noise(A):
	V = A.shape[0]
	#E = V * (V - 1) / 2
	p = 0.05 / V
	for vi in range(V):
		for vj in range(vi + 1, V):
			if p > np.random.random():
				filp_value = 1 - A[vi, vj]
				A[vi, vj] = A[vj, vi] = filp_value
	return A


def artificial_data(n_nodes, k_parts, isBM=True, isAP=False):
	M, C = artificial_MC(n_nodes, k_parts, isBM, isAP)
	A = artificial_A(M, C)
	#A = add_noise(A)
	labels = np.where(C == 1)[1]
	return A, labels


def syn_data(folder, node_nums, block_nums, isBM=True, isAP=False):
	for node_num in node_nums:
		for block_num in block_nums:
			adj_matrix, solutions = artificial_data(node_num, block_num, isBM, isAP)
			mat = {'adj_mat': adj_matrix, 'GT': solutions}
			if not isAP:
				filename = 'BM' if isBM else 'CD'
			else:
				filename = 'AP'
			filename += '_V' + str(node_num).zfill(4) + '_b' + str(block_num).zfill(2) + '.mat'
			savemat(folder + filename, mat)


def random_M(k_parts, p):
	M = np.zeros((k_parts, k_parts))
	for b1 in range(k_parts):
		for b2 in range(b1, k_parts):
			M[b1, b2] = M[b2, b1] = np.random.randint(1, 11) * p
	return M


def random_data(folder, node_nums, block_nums):
	for V in node_nums:
		for b in block_nums:
			membership = np.argmax(np.random.rand(V, b), axis=1)
			C = np.zeros((V, b))
			for v in range(V):
				C[v, membership[v]] = 1
			p = np.log(V) / V
			M = random_M(b, p)
			A = artificial_A(M, C)
			solutions = np.where(C == 1)[1]
			mat = {'adj_mat': A, 'GT': solutions}
			filename = 'V' + str(V).zfill(4) + 'b' + str(b).zfill(2) + '.mat'
			savemat(folder + filename, mat)


def louvain_CD(adj_matrix):
	start_time = time.time()
	G = nx.from_numpy_matrix(adj_matrix)
	partition = community.best_partition(G)
	runtime = time.time() - start_time
	labels = list(partition.values())
	obj = single_score(adj_matrix, labels)
	return labels, obj, runtime


def test_lovain(mat_files):
	for mat_file in mat_files:
		print(mat_file, end=' ')
		adj_matrix, clusters, solutions = load_mat(mat_file)
		labels, obj, runtime = louvain_CD(adj_matrix)
		print(obj, get_NMI(labels, solutions), str(runtime) + 's')


def graph_embedding_visual(G, P, labels, output='visual'):
	colors = ['r', 'b', 'g', 'm', 'y', 'c', 'orange', 'brown', 'cyan', 'lime', 'pink', 'yellow']
	nodes_color = []
	for label in labels:
		nodes_color.append(colors[label])
	nx.draw(G, node_color=nodes_color, node_size=50)
	plt.ion()
	plt.show()
	plt.savefig(f'{output}.png', dpi=300)
	plt.close()

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for v in range(P.shape[0]):
		ax.scatter(P[v, 0], P[v, 1], P[v, 2], color=nodes_color[v], marker='o', s=12)
	plt.ion()
	plt.savefig(f'{output}_fm.png', dpi=300)
	plt.close()


def test_matdata(times, mat_files, K, e, T, graphsets_num, isBM=True, dateset=''):
	results = []
	for mat_file in mat_files:
		print(mat_file)
		filename = mat_file[mat_file.rfind('/') + 1: mat_file.rfind('.')]
		adj_matrix, clusters, solutions = load_mat(mat_file)
		ave_obj, ave_NMI, ave_time = 0, 0, 0
		for t in range(times):
			print(t, datetime.datetime.now())
			labels, score, runtime, G, fm_embedding = bestscore_choice(adj_matrix, K, e, clusters, T, graphsets_num, isBM)
			NMI = get_NMI(labels, solutions)
			ave_obj, ave_NMI, ave_time = ave_obj + score, ave_NMI + NMI, ave_time + runtime
		ave_obj, ave_NMI, ave_time = ave_obj / times, ave_NMI / times, ave_time / times
		results.append([filename, round(ave_obj, 2), round(ave_NMI, 4), round(ave_time, 2)])
		print(results[-1])
		#graph_embedding_visual(G, fm_embedding, solutions, f'visualization/{dateset}/{filename}')
	df = pd.DataFrame(results)
	df.columns = ['Test Case', 'Objective', 'NMI', 'Time']
	print(df.to_latex(index=False))


if __name__ == '__main__':
	dateset = 'random'
	mat_files = glob.glob(f'datasets/{dateset}/*.mat')
	mat_files.sort()

	K, e, T, M, isBM = 4, 0.0001, 10, 4, True
	times, isAP = 10, True

	#adj_matrix = load_airdata('USAir97/air_adj.txt')
	#syn_data("datasets/sup_linear/", [400, 800, 1600, 3200], [10, 20], isBM, isAP)
	#test_lovain(mat_files)


	#random_data("datasets/random/", [400, 800, 1600, 3200], [4, 10, 20])

	test_matdata(times, mat_files, K, e, T, M, isBM, dateset)
