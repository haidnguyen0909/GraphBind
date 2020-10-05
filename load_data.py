import numpy as np
import csv
import scipy.sparse
import sys
import os
from os import path
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.backends.backend_pdf import PdfPages


SIZE = 10


def visualize_dna(prefix):
	file_reps = prefix + "_reps"
	file_labels = prefix + "_labels"
	file_masks = prefix + "_masks"
	reps_tmp = np.load(file_reps+'.npy')
	labels_tmp = np.load(file_labels+'.npy')
	masks = np.load(file_masks+'.npy')

	reps=[]
	labels = []
	
	for i in range(len(masks)):
		if i <= 1000:
			continue
		if masks[i][0] == 1.0:
			reps.append(reps_tmp[i,:])
			labels.append((labels_tmp[i][0]))
		else:
			reps.append(reps_tmp[i,:])
			labels.append(-1)
		#	print(labels[-1])
	

	print(len(reps))
	print(labels)

	feats = TSNE(n_components =2).fit_transform(reps)
	pdf = PdfPages(prefix+".pdf")
	cls=np.unique(labels)
	fea_num = [feats[labels==i] for i in cls]
	for i, f in enumerate(fea_num):
		if cls[i] == -1:
			plt.scatter(f[:,0],f[:,1],label=cls[i],marker='o')
		else:
			plt.scatter(f[:,0],f[:,1],label=cls[i],marker='+')
	plt.tight_layout()
	pdf.savefig()
	plt.show()
	pdf.close()


#prefix = 'vis3'


#visualize_dna(prefix)

def generateDic(seqs, size):
	dic = {}
	inv_dic = {}
	for seq, label in seqs:
		seqlen = len(seq)
		subseqs = [seq[i:i+size] for i in range(seqlen - size + 1)]
		for subseq in subseqs:
			if subseq not in dic.keys():
				dic[subseq] = len(dic)
				inv_dic[len(inv_dic)] = subseq
	return dic, inv_dic



def generateBOW(seqs, dic, size):
	n = len(seqs)
	d = len(dic)

	bow = np.zeros([n, d])
	labels = np.zeros(n)
	for i in range(n):
		seq = seqs[i][0]
		label = seqs[i][1]
		labels[i] = label
		for j in range(len(seq) - size + 1):
			subseq = seq[j:j+size]
			idx = dic[subseq]
			bow[i,idx] += 1.0
	return bow, labels


def load_data1(dataset, positive=True):
	if positive:
		data = "./data/" + dataset + "/"  + "positive.fasta"
	else:
		data = "./data/" + dataset + "/"  + "negative.fasta"
	f = open(data, "r")
	lines = f.readlines()
	#print(lines)
	n = len(lines)
	seqs = []
	for i in range(0, 50000, 2):
		str1 = lines[i].strip()
		str1 = str1.split("_")
		id = int(str1[-1])
		seq = lines[i+1].strip()
		if positive:
			label = 1.0
		else:
			label = 0.0
		seqs.append([seq, label])
	return seqs
		

def load_data(dataset):
	data = "./data/" + dataset + "/" + dataset + "_Data.txt"
	ratio = "./data/" + dataset + "/" + dataset +"_Ratio.txt"
	seqs = []
	with open(data) as f:
		reader = csv.reader(f, delimiter = '\t')
		headline = True
		for row in reader:
			if headline:
				headline = False
				continue
			seq = row[0]
			label = float(row[1])
			#print(row)
			#print(seq, label)
			seqs.append([seq, label])
	return seqs

def load_data2(dataset):# for loading MITO datasets
	data = "./data/" + dataset + "/" + dataset + "_Data.txt"
	ratio = "./data/" + dataset + "/" + dataset +"_Ratio.txt"
	seqs = []
	with open(data) as f:
		reader = csv.reader(f, delimiter = '\t')
		headline = True
		for row in reader:
			if headline:
				headline = False
				continue
			seq = row[1]
			label = float(row[-1])
			if np.isnan(label) or len(seq)!=52:
				continue
			#print(row)
			#print(seq, label)
			seqs.append([seq, label])
	uniques = {}
	for seq, label in seqs:
		if seq not in uniques.keys():
			uniques[seq] =[label]
		else:
			tmp = uniques[seq]
			tmp.append(label)
			uniques[seq] = tmp

	seqs=[]
	for seq in uniques.keys():
		label = np.mean(uniques[seq])
		seqs.append([seq, label])
		#print(seq, uniques[seq])
	#print(len(seqs))

	return seqs

def load_data_HITSFLIP(dataset):
	data = "./data/" + dataset + "/" + dataset + "_Data.txt"
	seqs=[]
	with open(data) as f:
		reader = csv.reader(f, delimiter=" ")
		for row in reader:
			seq = row[0]
			label = float(row[-1])
			seqs.append([seq, label])
	return seqs

#load_data1("test1")

def load_DREAM_chipset(dataset):
	seqs = []
	with open(dataset) as f:
		reader = csv.reader(f, delimiter = '\t')
		for row in reader:
			seq = row[0]
			label = float(row[1])
			seqs.append([seq, label])

	return seqs		


def load_tfgroups():
	tfids = []
	with open('./data/pbm/tfids.txt') as f:
		for line in f.readlines():
			line = line.rstrip("\r\n")
			tfids.append(line)
	tfgroups = []
	fold1 = {}
	fold1['ids'] = set(tfids).intersection(["TF_%d"%(i+1) for i in range(0,33)])
	fold1['train_fold'] = 'A'
	fold1['test_fold'] = 'B'

	fold2 = {}
	fold2['ids'] = set(tfids).intersection(["TF_%d"%(i+1) for i in range(33,66)])
	fold2['train_fold'] = 'B'
	fold2['test_fold'] = 'A'

	tfgroups.append(fold1)
	tfgroups.append(fold2)
	return tfgroups


def load_probe_biases():
	#if not os.path.exist('./data/pbm/probe_biases.npz'):
	targets=[]
	header = True
	with open('./data/pbm/targets.tsv') as f:
		for line in f.readlines():
			if header:
				header = False
				continue
			row = line.rstrip().split('\t')
			#print(row)
			targets.append([float(e) for e in row])

	targets = np.array(targets)
	bias = []
	for i in range(len(targets)):
		measurement = targets[i,:].ravel()
		if np.all(np.isnan(measurement)):
			bias.append(np.nan)
		else:
			tmp = ~np.isnan(measurement)
			median = np.median(measurement[tmp])
			bias.append(median)
	return bias

def load_name2id():
	with open('./data/pbm/targets.tsv') as f:
		targetnames = f.readline().rstrip().split('\t')
	name2id = {}
	for id, targetname in enumerate(targetnames):
		name2id[targetname] = id
	return name2id

def generate_tag(ids, trn_fold, tst_fold):
	tag = np.zeros(len(ids))
	#1: training, 2: valid, 3: test
	train_ids=[]
	for i in range(len(tag)):
		if ids[i] == tst_fold:
			tag[i] = 3
		else:
			tag[i] = 1
			train_ids.append(i)
	n_train = len(train_ids)
	n_valid = int(n_train/10)-1
	for j in range(0, n_train,10):
		val_id = train_ids[j]
		tag[val_id] = 2
	return tag


def load_pbmdata(tfid, remove_bias=False, limit = 1000):
	
	if remove_bias:
		bias = load_probe_biases()
	sequences = []
	targets =[]
	foldids=[]
	header = True
	with open('./data/pbm/sequences.tsv') as f:
		for line in f.readlines():
			if header:
				header = False
				continue
			line = line.rstrip().split('\t')
			foldids.append(line[0])
			sequences.append(line[-1])

	header = True
	with open('./data/pbm/targets.tsv') as f:
		for line in f.readlines():
			if header:
				header = False
				continue
			line = line.rstrip().split('\t')
			targets.append(float(line[tfid]))

	# filtering sequences with nan values
	rsequences = []
	rfoldids = []
	count={}
	count['A'] = 0
	count['B'] = 0
	print("\n filtering")
	for i in range(len(sequences)):
		if np.isnan(targets[i]):
			continue
		if count[foldids[i]] > limit:
			continue

		rfoldids.append(foldids[i])
		count[foldids[i]]+=1
		if remove_bias:
			tmp = targets[i]/bias[i]
		else:
			tmp = targets[i]
		rsequences.append([sequences[i], tmp])


	return rfoldids, rsequences

def dinucshuffle(sequence):
	b = [sequence[i:i+2] for i in range(0, len(sequence), 2)]
	np.random.shuffle(b)
	d = ''.join([str(x) for x in b])
	return d


def modifystr(str, size=0):
  n = len(str)
  return str[size:n-size]
def reverse(seq):
	dictReverse={'A':'T','C':'G','G':'C','T':'A','N':'N'}
	rseq = [dictReverse[seq[c]] for c in range(len(seq))]
	rseq = ''.join(rseq)
	#print(seq, rseq)
	return rseq


def generate_encode(tfid):
	seq_suffix =".seq"
	filename = "./data/encode/%s_B%s" % (tfid, seq_suffix)
	p_sequences = []
	n_sequences = []
	targets=[]
	header=True
	count=0
	with open(filename) as f:
		for line in f.readlines():
			if header:
				header=False
				continue
			count+=1
			line = line.rstrip().split('\t')
			seq = line[2]
			target=float(line[-1])
			nseq = dinucshuffle(seq)
			p_sequences.append([seq, 1.0])
			n_sequences.append([nseq, 0.0])

	#print(n_sequences)
	#print(len(p_sequences), len(n_sequences))
	with open(filename, "a") as f:
		print("writing file")
		for n_seq in n_sequences:
			seq = n_seq[0]
			target = n_seq[1]
			line = "\n" + 'A\t' + "shuff\t" + seq + "\t" + str(target)
			f.write(line)
	return p_sequences+n_sequences





def load_encode_train(tfid, limit=500, use_reverse=False):
	seq_suffix =".seq"
	filename = "./data/encode/%s_AC%s" % (tfid, seq_suffix)
	sequences=[]
	targets=[]
	header=True
	count=0
	tmp=[]
	with open(filename) as f:
		for line in f.readlines():
			if header:
				header=False
				continue
			count+=1
			if count >= limit:
				break
			line = line.rstrip().split('\t')
			seq = line[2]
			seq = modifystr(seq, SIZE)
			target = float(line[-1])
			if use_reverse:
				seq = reverse(seq)
			#nseq = reverse(dinucshuffle(seq))
			nseq = dinucshuffle(seq)
			sequences.append([seq, 1])
			sequences.append([nseq, 0])
			#sequences.append([dinucshuffle(seq), 0])		
	return sequences

def load_encode_test(tfid, use_reverse=False):
	seq_suffix =".seq"
	filename = "./data/encode/%s_B%s" % (tfid, seq_suffix)
	sequences=[]
	targets=[]
	header=True
	with open(filename) as f:
		lines = f.readlines()
		for line in lines:
			if header:
				header=False
				continue
			line = line.rstrip().split('\t')
			if len(line) < 4:
				continue
			seq = line[2]
			seq = modifystr(seq, SIZE)
			if use_reverse:
				seq = reverse(seq)
			target = float(line[-1])
			sequences.append([seq, target])

	return sequences


#tfid="ARID3A_K562_ARID3A_(sc-8821)_Stanford"
#load_encode_train(tfid)
#X, Y = load_pbmdata(['TF_1', 'TF_2'], 'A', remove_bias = True)
#print(X, Y)
#print(len(X), len(Y))


#biases = load_probe_biases()
#print(biases)

#load_DREAM_chipset('./data/chipseq/TF_23_CHIP_51_full_genomic.seq.txt')
#if len(sys.argv) != 2:
#	sys.exit("Use: python load_data.py <dataset_name>")

# hyperparameters
#kmer_size = 3

#dataset = sys.argv[1]
#data = "./data/" + dataset + "/" + dataset + "_Data.txt"
#ratio = "./data/" + dataset + "/" + dataset +"_Ratio.txt"

# reading data




#dic, inv_dic = generateDic(seqs, kmer_size)
#save_dic = "./data/" + dataset + "/" + dataset + "_dic.npy"
#np.save(save_dic, dic)
#save_seqs = "./data/" + dataset + "/" + dataset + "_seqs.npy"
#np.save(save_seqs, seqs)

#seqs_t = np.load(save_seqs)
#print(seqs_t)


#bow, labels = generateBOW(seqs, dic, kmer_size)

# save bow, labels and dic
#save_bow = "./data/" + dataset + "/" + dataset + "_bow.npz"
#save_labels = "./data/" + dataset + "/" + dataset + "_labels.npy"
#save_dic = "./data/" + dataset + "/" + dataset + "_dic.npy"

#sparse_bow = scipy.sparse.csc_matrix(bow)

#scipy.sparse.save_npz(save_bow, sparse_bow)
#np.save(save_labels, labels)
#np.save(save_dic, dic)

#sparse_bow_t = scipy.sparse.load_npz(save_bow)
#labels_t = np.load(save_labels)
#dic_t = np.load(save_dic).item()










