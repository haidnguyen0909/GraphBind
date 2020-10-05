import numpy as np
import csv
import scipy.sparse
import sys
import Levenshtein
from scipy.sparse import hstack
from scipy.sparse import vstack


def priorfeature(seqs, position_encoding, adj, size):

	word_vocab = {}
	n_sqs = len(seqs)
	for i, seq_label in enumerate(seqs):
		seq = seq_label[0]
		label = seq_label[1]
		subseqs = [seq[i:i+size] for i in range(len(seq) - size + 1)]
		for subseq in subseqs:
			if subseq not in word_vocab.keys():
				word_vocab[subseq] = len(word_vocab)
	# not we have word_vocab & position_encoding
	# we need to calculate prior features for only doc
	n_pos, n_dim = position_encoding.shape
	print("position encoding dim: %d x %d" % (n_pos, n_dim))
	F = np.zeros((n_sqs, n_dim))
	for i, seq_label in enumerate(seqs):
		seq = seq_label[0]
		label = seq_label[1]
		subseqs = [seq[i:i+size] for i in range(len(seq) - size + 1)]
		for pos, subseq in enumerate(subseqs):
			w_id = word_vocab[subseq] + n_sqs
			pos_enc = position_encoding[pos, :]
			weight = adj[i, w_id]
			F[i,:] += weight * pos_enc
	return F

def concatenate(A, B, sizeA, sizeB):
	sample_size = A.shape[0] - sizeA
	part1  = A
	part21 = B[sample_size:sample_size+sizeB,0:sample_size]
	part22 = scipy.sparse.csr_matrix((sizeB, sizeA))
	part2 = hstack((part21, part22))
	part12 = vstack((part1, part2))


	part31 = B[0:sample_size, sample_size:sample_size+sizeB]
	part32 = scipy.sparse.csr_matrix((sizeA, sizeB))
	part33 = B[sample_size:sample_size+sizeB,sample_size:sample_size+sizeB]
	part312 = vstack((part31, part32))
	part3 = vstack((part312, part33))

	return hstack((part12, part3)), sizeA+sizeB

def reverse(seq):
	dictReverse={'A':'T','C':'G','G':'C','T':'A','N':'N'}
	rseq = [dictReverse[seq[c]] for c in range(len(seq))]
	rseq = ''.join(rseq)
	return rseq

def concatenate_all(s_list): # [[A, sizeA],[B, sizeB],...]
	initial =s_list[0]
	s_A = initial[0]
	size_A = initial[1]
	if len(s_list) == 1:
		return s_A, size_A
	for i in range(1, len(s_list)):
		s_B = s_list[i][0]
		size_B = s_list[i][1]
		s_A, size_A = concatenate(s_A, s_B, size_A, size_B)
	return s_A, size_A



def label_folds(n, k):
	tag = [0] *n
	for i in range(n):
		tag[i] = i%k + 1
	#return np.random.permutation(np.array(tag))
	return np.array(tag)

def construct_inv_vocab(seqs, size, thresh = 3):
	word_inv_vocab = {}
	word_vocab={}
	word_freq = {}
	n_sqs = len(seqs)
	for i, seq_label in enumerate(seqs):
		seq= seq_label[0]
		subseqs = [seq[i:i+size] for i in range(0, len(seq) - size + 1, 1)]
		for subseq in subseqs:
			if subseq not in word_freq:
				word_freq[subseq] = 1
			else:
				word_freq[subseq]+=1
	#print(word_freq)
	for word in word_freq.keys():
		freq = word_freq[word]
		if freq < thresh:
			continue

		word_vocab[word] = len(word_vocab)
		word_inv_vocab[len(word_inv_vocab)] = word
	print("size of vocabulary with %d-mer is %d" %(size, len(word_vocab)))
	#print(word_freq)
	return word_inv_vocab



def construct_vocab(seqs, size, thresh = 3):
	word_vocab = {}
	word_freq = {}
	#word_importance = {}
	n_sqs = len(seqs)
	for i, seq_label in enumerate(seqs):
		seq= seq_label[0]
		label = seq_label[1]
		subseqs = [seq[i:i+size] for i in range(0, len(seq) - size + 1, 1)]
		for subseq in subseqs:
			if subseq not in word_freq:
				word_freq[subseq] = 1
			else:
				word_freq[subseq]+=1
			#if subseq not in word_importance:
			#	word_importance[subseq] = [label]
			#else:
			#	tmp = word_importance[subseq]
			#	tmp.append(label)
			#	word_importance[subseq] = tmp
			#rev_subseq = reverse(subseq)
			#if rev_subseq not in word_freq:
			#	word_freq[rev_subseq] = 1
			#else:
			#	word_freq[rev_subseq] +=1
	#print(word_freq)
	for word in word_freq.keys():
		#cum_val = np.mean(word_importance[word])
		#print(cum_val)
		#if cum_val <= thresh:
		#	continue
		freq = word_freq[word]
		if freq < thresh:
			continue
		word_vocab[word] = len(word_vocab)
	print("size of vocabulary with %d-mer is %d" %(size, len(word_vocab)))
	#print(word_freq)
	return word_vocab

def reverse_vocab(word_vocab):
	rev_word_vocab = {}
	for key in word_vocab.keys():
		rkey = reverse(key)
		rev_word_vocab[rkey] = word_vocab[key]
	return rev_word_vocab

def rev_w2w_local(seqs, rev_seqs, size_list, local_weight):
	adjs=[]
	rev_adjs=[]
	for size in size_list:
		word_vocab = construct_vocab(seqs, size)
		adj = construct_w2w_local(seqs, size, word_vocab, local_weight)
		adjs.append([adj, len(word_vocab)])

		rev_word_vocab = reverse_vocab(word_vocab)
		#rev_word_vocab = word_vocab
		rev_adj = construct_w2w_local(rev_seqs, size, rev_word_vocab, local_weight)
		#rev_adj = adj
		rev_adjs.append([rev_adj, len(rev_word_vocab)])

	final_adj, final_size = concatenate_all(adjs)
	final_rev_adj, final_size = concatenate_all(rev_adjs)
	#final_rev_adj = final_adj
	return final_adj, final_rev_adj

def w2w_local(seqs, size_list, local_weight):
	word_vocab_list = []
	
	adjs=[]
	for size in size_list:
		word_vocab = construct_vocab(seqs, size)
		adj = construct_w2w_local(seqs, size, word_vocab, local_weight)
		adjs.append([adj, len(word_vocab)])
	final_adj, final_size = concatenate_all(adjs)
	return final_adj

def rev_w2w_global(seqs, rev_seqs, size_list):
	adjs=[]
	rev_adjs=[]
	for size in size_list:
		word_vocab = construct_vocab(seqs, size)
		adj = construct_w2w_global(seqs, size, word_vocab)
		adjs.append([adj, len(word_vocab)])

		rev_word_vocab = reverse_vocab(word_vocab)
		rev_adj = construct_w2w_global(rev_seqs, size, rev_word_vocab)
		#rev_word_vocab = word_vocab
		#rev_adj = adj
		rev_adjs.append([rev_adj, len(rev_word_vocab)])

	final_adj, final_size = concatenate_all(adjs)
	final_rev_adj, final_size = concatenate_all(rev_adjs)

	return final_adj, final_rev_adj

def w2w_global(seqs, size_list):
	word_vocab_list = []
	adjs=[]
	for size in size_list:
		word_vocab = construct_vocab(seqs, size)
		adj = construct_w2w_global(seqs, size, word_vocab)
		adjs.append([adj, len(word_vocab)])
	final_adj, final_size = concatenate_all(adjs)
	return final_adj

def rev_d2w_global(seqs, rev_seqs, size_list):
	adjs=[]
	rev_adjs=[]
	for size in size_list:
		word_vocab = construct_vocab(seqs, size)
		adj = construct_d2w_global(seqs, size, word_vocab)
		adjs.append([adj, len(word_vocab)])
		rev_word_vocab = reverse_vocab(word_vocab)
		#rev_word_vocab = word_vocab

		rev_adj = construct_d2w_global(rev_seqs, size, word_vocab)
		rev_adjs.append([rev_adj, len(rev_word_vocab)])
		#rev_adj = adj
		rev_adjs.append([rev_adj, len(rev_word_vocab)])
	final_adj, final_size = concatenate_all(adjs)
	final_rev_adj, final_size = concatenate_all(rev_adjs)
	return final_adj, final_rev_adj

def d2w_global(seqs, size_list):
	adjs=[]
	for size in size_list:
		word_vocab = construct_vocab(seqs, size)
		adj = construct_d2w_global(seqs, size, word_vocab)
		adjs.append([adj, len(word_vocab)])
	final_adj, final_size = concatenate_all(adjs)
	return final_adj


def construct_d2w_global(seqs, size, word_vocab):
	word_doc_list = {}
	n_sqs = len(seqs)
	#print(word_vocab)
	for i, seq_label in enumerate(seqs):
		seq = seq_label[0]
		label = seq_label[1]
		subseqs = [seq[i:i+size] for i in range(0, len(seq) - size + 1, 1)]
		for subseq in subseqs:
			if subseq not in word_vocab:
				continue
			#if subseq not in word_vocab.keys():
			#	word_vocab[subseq] = len(word_vocab)
			if subseq in word_doc_list:
				doc_list = word_doc_list[subseq]
				if i not in doc_list:
					doc_list.append(i)
				word_doc_list[subseq] = doc_list
			else:
				word_doc_list[subseq] = [i]
			'''
			rev_subseq = reverse(subseq)
			if rev_subseq not in word_vocab:
				continue
			if rev_subseq in word_doc_list:
				doc_list = word_doc_list[rev_subseq]
				if i not in doc_list:
					doc_list.append(i)
				word_doc_list[rev_subseq] = doc_list
			else:
				word_doc_list[rev_subseq] = [i]
			'''

	word_doc_freq = {}
	sum = 0.0
	for word, doc_list in word_doc_list.items():
		word_doc_freq[word] = len(doc_list)
		sum += word_doc_freq[word]


	doc_word_freq = {}
	for i, seq_label in enumerate(seqs):
		seq = seq_label[0]
		label = seq_label[1]
		subseqs = [seq[i:i+size] for i in range(0,len(seq) - size + 1, 1)]
		word_set = []
		for subseq in subseqs:
			if subseq not in word_vocab:
				continue
			word_id = word_vocab[subseq]
			if word_id not in word_set:
				word_set.append(word_id)
			doc_word_str = str(i) + ',' + str(word_id)
			if doc_word_str in doc_word_freq:
				doc_word_freq[doc_word_str] +=1
			else:
				doc_word_freq[doc_word_str] = 1
			'''
			rev_subseq = reverse(subseq)
			if rev_subseq not in word_vocab:
				continue
			word_id = word_vocab[rev_subseq]
			if word_id not in word_set:
				word_set.append(word_id)
			doc_word_str = str(i) + ',' + str(word_id)
			if doc_word_str in doc_word_freq:
				doc_word_freq[doc_word_str] +=1
			else:
				doc_word_freq[doc_word_str] = 1
			'''


	row =[]
	col=[]
	weight=[]

	for i, seq_label in enumerate(seqs):
		seq = seq_label[0]
		label = seq_label[1]
		subseqs = [seq[i:i+size] for i in range(0, len(seq) - size + 1, 1)]
		doc_word_set = set()
		for subseq in subseqs:
			if subseq not in word_vocab:
				continue
			if subseq in doc_word_set:
				continue
			doc_word_set.add(subseq)
			key = str(i)+',' +str(word_vocab[subseq])
			freq = doc_word_freq[key]/len(subseqs)
			idf = np.log(1.0 * len(seqs)/word_doc_freq[subseq])
			#print(word_vocab[subseq])
			w = freq * idf
			j = word_vocab[subseq]
			row.append(i)
			col.append(j + n_sqs)
			weight.append(w)
		'''
		for subseq in subseqs:
			rev_subseq = reverse(subseq)
			if rev_subseq not in word_vocab:
				continue
			if rev_subseq in doc_word_set:
				continue
			doc_word_set.add(rev_subseq)
			key = str(i)+',' +str(word_vocab[rev_subseq])
			freq = doc_word_freq[key]/len(rev_subseq)
			idf = np.log(1.0 * len(seqs)/word_doc_freq[rev_subseq])
			#print(word_vocab[subseq])
			w = freq * idf
			j = word_vocab[rev_subseq]
			row.append(i)
			col.append(j + n_sqs)
			weight.append(w)
			#weight.append(0.0)
		'''
			#print(row[-1],col[-1],weight[-1])
			#col.append(i)
			#row.append(j + n_sqs)
			#print(word_vocab[subseq])
			#weight.append(freq * idf)
			#print(row[-1],col[-1],weight[-1])
			#print(i,j,weight[-1])
	print("MIN WIEGHT: %f, max: %f"%(np.min(weight), np.max(weight)) )
	vocab_size = len(word_vocab)
	node_size = len(seqs) + vocab_size
	adj = scipy.sparse.csr_matrix((weight, (row, col)),shape=(node_size, node_size))
	adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
	return adj




def construct_w2w_local(seqs, size, word_vocab, local_weight):

	row=[]
	col=[]
	weight=[]
	#print(word_vocab)
	vocab_size = len(word_vocab)
	dis = np.zeros((vocab_size, vocab_size))
	for word1 in word_vocab.keys():
		for word2 in word_vocab.keys():
			if word1==word2:
				continue
			id1 = word_vocab[word1]
			id2 = word_vocab[word2]
			dis[id1, id2] = Levenshtein.distance(word1, word2)
			dis[id2, id2] = dis[id1, id2]
			#row.append(id1 + len(seqs))
			#col.append(id2 + len(seqs))
			#weight.append(np.exp(-0.8 * dis))
			#print(row[-1],col[-1],weight[-1])
	#print(dis)
	nnb = 3 * size
	#print(word_vocab)
	for word in word_vocab.keys():
		id = word_vocab[word]
		distances = dis[id, :]
		distances[id] = 1000
		kNbrs = np.argpartition(distances, nnb)
		kNbrs = kNbrs[:nnb]
		#print(word, kNbrs, distances[kNbrs])
		for Nbr in kNbrs:
			row.append(id + len(seqs))
			col.append(Nbr + len(seqs))
			weight.append(local_weight)
	node_size = len(seqs) + vocab_size
	adj = scipy.sparse.csr_matrix((weight, (row, col)),shape=(node_size, node_size))
	adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
	return adj



def construct_w2w_global(seqs, size, word_vocab):
	word_freq = {}
	n_sqs = len(seqs)
	n_w = 0
	for seq, label in seqs:
		#print(seq)
		subseqs = [seq[i:i+size] for i in range(len(seq) - size + 1)]
		for subseq in subseqs:
			if subseq not in word_vocab:
				continue
			#n_w +=1
			#if subseq not in word_vocab:
			#	word_vocab[subseq] = len(word_vocab)

	window_size = 40
	windows = []
	for seq, label in seqs:
		length = len(seq)
		if length <= window_size:
			windows.append(seq)
		else:
			for j in range(length - window_size + 1):
				window = seq[j:j+window_size]
				windows.append(window)

	for window in windows:
		appeared = set()
		for i in range(len(window) - size + 1):
			s_i = window[i:i+size]
			if s_i not in word_vocab:
				continue
			#if s_i in appeared:
			#	continue
			if s_i in word_freq:
				word_freq[s_i] +=1
			else:
				word_freq[s_i] = 1
			appeared.add(s_i)

	word_pair_count = {}
	n_p = 0
	for window in windows:
		for i in range(len(window) - size + 1):
			s_i = window[i:i+size]
			if s_i not in word_vocab:
				continue
			for j in range(i+1, len(window) - size + 1):
				
				s_j = window[j:j+size]
				if s_j not in word_vocab:
					continue
				if s_i == s_j:
					continue
				word_pair_str = s_i + "," + s_j
				if word_pair_str in word_pair_count:
					word_pair_count[word_pair_str] +=1
				else:
					word_pair_count[word_pair_str] =1
				word_pair_str = s_j + "," + s_i
				if word_pair_str in word_pair_count:
					word_pair_count[word_pair_str] +=1
				else:
					word_pair_count[word_pair_str] =1

	pos_count = 0
	neg_count = 0
	row = []
	col = []
	weight = []
	vocab_size = len(word_vocab)
	dis = np.zeros((vocab_size, vocab_size))

	n_windows = len(windows)

	for key in word_pair_count.keys():
		temp = key.split(',')
		count = word_pair_count[key]
		i = temp[0]
		j = temp[1]
		id1 = word_vocab[i]
		id2 = word_vocab[j]
		w_freq_i = word_freq[i]
		w_freq_j = word_freq[j]
		#pmi = np.log((1.0*count*n_windows)/((w_freq_i*w_freq_j)))
		pmi = 1.0*count/(w_freq_i*1.0)
		#if pmi < 0:
		#	continue
		dis[id1, id2] = pmi
		

	nnb = 10

	for word in word_vocab.keys():
		index_i = word_vocab[word]
		distances = -dis[index_i, :]
		distances[index_i] = 1000
		kNbrs = np.argpartition(distances, nnb)
		kNbrs = kNbrs[:nnb]
		#dis[index_i, index_j] = -count/sum2
		#dis[index_j, index_i] = -count/sum2
		for Nbr in kNbrs:
			if dis[index_i, Nbr] <= 0:
				continue
			row.append(index_i + len(seqs))
			col.append(Nbr + len(seqs))
			weight.append(dis[index_i, Nbr])
			#print(row[-1], col[-1], weight[-1])
		
		
	#nnb = 5
	#print(word_vocab)
	#for word in word_vocab.keys():
	#	id = word_vocab[word]
	#	distances = dis[id, :]
	#	distances[id] = 1000
		#print(distances)
	#	kNbrs = np.argpartition(distances, nnb)
	#	kNbrs = kNbrs[:nnb]
	#	sum = np.sum(dis[id, kNbrs])


		
	#	for Nbr in kNbrs:
	#		if sum == 0:
	#			continue
	#		row.append(id)
	#		col.append(Nbr)
	#		weight.append(-dis[id, Nbr])
		#	weight.append(1)
	#		print(id, Nbr, -dis[id, Nbr])
			

	node_size = len(seqs) + vocab_size
	adj = scipy.sparse.csr_matrix((weight, (row, col)),shape=(node_size, node_size))
	#adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
	return adj
	




def construct_w2w_global1(seqs, size, word_vocab):
	#print("DKMDKM")
	word_freq = {}
	n_sqs = len(seqs)
	n_w = 0


	window_size = 50
	windows = []
	for seq, label in seqs:
		length = len(seq)
		if length <= window_size:
			windows.append(seq)
		else:
			for j in range(0, length - window_size + 1, 1):
				window = seq[j:j+window_size]
				windows.append(window)
	print("DKM1")
	for window in windows:
		appeared = set()
		for i in range(0, len(window) - size + 1, 1):
			s_i = window[i:i+size]
			if s_i not in word_vocab:
				continue
			if s_i in word_freq:
				word_freq[s_i] +=1
			else:
				word_freq[s_i] = 1
			appeared.add(s_i)

	print("DKM2")

	word_pair_count = {}
	n_p = 0
	for window in windows:
		for i in range(len(window) - size + 1):
			s_i = window[i:i+size]
			if s_i not in word_vocab:
				continue
			for j in range(i+1, len(window) - size + 1):
				
				s_j = window[j:j+size]
				if s_j not in word_vocab:
					continue
				if s_i == s_j:
					continue
				word_pair_str = s_i + "," + s_j
				if word_pair_str in word_pair_count:
					word_pair_count[word_pair_str] +=1
				else:
					word_pair_count[word_pair_str] =1
				word_pair_str = s_j + "," + s_i
				if word_pair_str in word_pair_count:
					word_pair_count[word_pair_str] +=1
				else:
					word_pair_count[word_pair_str] =1
	#print(word_pair_count)
	print("DKM3")
	row = []
	col = []
	weight = []
	vocab_size = len(word_vocab)
	n_windows = len(windows)

	for key in word_pair_count.keys():
		temp = key.split(',')
		count = word_pair_count[key]
		i = temp[0]
		j = temp[1]
		id1 = word_vocab[i]
		id2 = word_vocab[j]
		w_freq_i = word_freq[i]
		w_freq_j = word_freq[j]
		pmi = np.log((1.0*count*n_windows)/((w_freq_i*w_freq_j)))
		if pmi < 0.0000:
			continue
		row.append(id1 + len(seqs))
		col.append(id2 + len(seqs))
		pmi = 1.0*count/(w_freq_i*1.0)
		weight.append(pmi)
		#print(row[-1], col[-1], weight[-1])
		
	print("max: %f, min:%f" %(np.max(weight),np.min(weight)))
	node_size = len(seqs) + vocab_size
	adj = scipy.sparse.csr_matrix((weight, (row, col)),shape=(node_size, node_size))
	#adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
	return adj
	









# save bow, labels and dic

#save_seqs = "./data/" + dataset + "/" + dataset + "_seqs.npy"
#save_dic = "./data/" + dataset + "/" + dataset + "_dic.npy"


#dic = np.load(save_dic).item()
#seqs = np.load(save_seqs)

#np.random.shuffle(seqs)
#tag = label_folds(len(seqs), 10)

#test_ids = np.where(tag == 1)
#train_ids = np.where(tag != 1)
#ids = train_ids + test_ids





#word_pair_count = {}
#n_p = 0
#for seq, label in seqs:
#	for i in range(len(seq) - size + 1):
#		for j in range(i, len(seq) - size + 1):
#			n_p +=1
#			s_i = seq[i: i+size]
#			s_j = seq[j: j+size]

#			word_pair_str = s_i + "," + s_j
#			if word_pair_str in word_pair_count:
#				word_pair_count[word_pair_str] +=1
#			else:
#				word_pair_count[word_pair_str] =1

#			word_pair_str = s_j + "," + s_i
#			if word_pair_str in word_pair_count:
#				word_pair_count[word_pair_str] +=1
#			else:
#				word_pair_count[word_pair_str] =1




# construct graph
#row = []
#col = []
#weight = []

#nzero = 0
#ntotal = 0
#for key in word_pair_count.keys():
#	temp = key.split(',')
#	count = word_pair_count[key]
#	i = temp[0]
#	j = temp[1]
#	w_freq_i = word_freq[i]
#	w_freq_j = word_freq[j]
#	pmi = np.log(count * sum1 * sum1/(w_freq_i*w_freq_j * sum2))
#	if pmi <= 0:
#		continue
#	row.append(word_vocab[i])
#	col.append(word_vocab[j])
#	weight.append(pmi)
#

#word_doc_list = {}

#for i, seq, label in enumerate(seqs):
#	subseqs = [seq[i:i+size] for i in range(len(seq) - size + 1)]
#	for subseq in subseqs:
#		if subseq in word_doc_list:
#			doc_list = word_doc_list[subseq]
#			doc_list.append(i)
#		else:
#			word_doc_list[subseq] = [i]
#word_doc_freq = {}
#for word, doc_list in word_doc_list.items():
#	word_doc_freq[word] = len(doc_list)

#doc_word_freq = {}

#for i, seq, label in enumerate(seqs):
#	subseqs = [seq[i:i+size] for i in range(len(seq) - size + 1)]
#	for subseq in subseqs:
#		word_id = word_vocab[subseq]
#		doc_word_str = str(i) + ',' + str(word_id)
#		if doc_word_str in doc_word_freq:
#			doc_word_freq[doc_word_str] +=1
#		else:
#			doc_word_freq[doc_word_str] = 1
#for i, seq, label in enumerate(seqs):
#	subseqs = [seq[i:i+size] for i in range(len(seq) - size + 1)]
#	doc_word_set = set()
#	for subseq in subseqs:
#		if subseq in doc_word_set:
#			continue
#		key = str(i)+',' +str(word_vocab[subseq])
#		freq = doc_word_freq[key]
#		row.append(i + vocab_size)
#		col.append(word_vocab[subseq])
#		idf = log(1.0 * len(seqs)/word_doc_freq[word_vocab[subseq]])
#		weight.append(freq * idf)
#











#node_size = train_size + vo
#adj = scipy.sparse.csr_matrix((weight, (row, col)),shape=(node_size, node_size))
	














