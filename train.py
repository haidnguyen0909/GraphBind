import tensorflow as tf  
import sys
import os
import load_data
from load_data import *
import construct_local_graph
from construct_local_graph import *
from utils import *
import models
from models import GCN, dGCN
from sklearn import metrics
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def discovermotif(scores, vocab, k=100):
	scores = np.array(scores)
	inds = np.argpartition(scores, -k)[-k:]
	seqs = []
	for ind in inds.tolist():
		print(ind, (scores[ind]), vocab[ind])
		seqs.append(vocab[ind])
	return seqs

SIZE = [10]




def generateAdjs(tfids):
	for tfid in tfids:
		print("PROCESSING %s" % tfid)
		train_seqs = load_encode_train(tfid)
		test_seqs = load_encode_test(tfid)
		rev_train_seqs = load_encode_train(tfid, use_reverse=True)
		rev_test_seqs = load_encode_test(tfid, use_reverse=True)
		seqs = train_seqs + test_seqs
		rev_seqs = rev_train_seqs + rev_test_seqs
		sizes = [8,10]
		ext = ".npz"
		strsize =""
		for size in sizes:
			strsize+=str(size)
		filename = "./adjs/"+tfid + "_" + strsize + "_" + "adj3" + ext
		rev_filename = "./adjs/"+tfid + "_" + strsize + "_" + "_rev_adj3" + ext
		#if os.path.exists(filename):
		#	adj3 = scipy.sparse.load_npz(filename)
		#	rev_adj3 = scipy.sparse.load_npz(rev_filename)
		#	print("load " + filename)
		#else:
		if True:
			adj3, rev_adj3 = rev_d2w_global(seqs, rev_seqs, sizes)
			#adj3 = d2w_global(seqs, sizes)
			scipy.sparse.save_npz(filename, adj3)
			scipy.sparse.save_npz(rev_filename, rev_adj3)
			print("save " + filename)
		filename = "./adjs/"+tfid + "_" + strsize + "_" + "adj2" + ext
		rev_filename = "./adjs/"+tfid + "_" + strsize + "_" + "_rev_adj2" + ext
		#if os.path.exists(filename):
		#	adj2 = scipy.sparse.load_npz(filename)
		#	rev_adj2 = scipy.sparse.load_npz(rev_filename)
		#	print("load " + filename)
		if True:
			#adj2 = w2w_local(seqs, sizes, 1.0)
			adj2, rev_adj2 = rev_w2w_local(seqs, rev_seqs, sizes, 1.0)
			scipy.sparse.save_npz(filename, adj2)
			scipy.sparse.save_npz(rev_filename, rev_adj2)
			print("save " + filename)
		filename = "./adjs/"+tfid + "_" + strsize + "_" + "adj1" + ext
		rev_filename = "./adjs/"+tfid + "_" + strsize + "_" + "_rev_adj1" + ext
		#if os.path.exists(filename):
		#	adj1 = scipy.sparse.load_npz(filename)
		#	rev_adj1 = scipy.sparse.load_npz(rev_filename)
		#	print("load " + filename)
		#else:
		if True:
			#adj1 = w2w_global(seqs, sizes)
			adj1, rev_adj1=rev_w2w_global(seqs, rev_seqs, sizes)
			scipy.sparse.save_npz(filename, adj1)
			scipy.sparse.save_npz(rev_filename, rev_adj1)
			print("save " + filename)


def calc_tpc_fpc_curve(z, y):
	order = np.argsort(z, axis=0, kind="mergesort")[::-1].ravel()
	z = z[order]
	y = y[order]
	# Accumulate the true positives with decreasing threshold
	tpc = y.cumsum()
	fpc = 1 + np.arange(len(y)).astype(y.dtype) - tpc
	return tpc, fpc
def calc_auc(z, y, want_curve=False):
	tpc, fpc = calc_tpc_fpc_curve(z, y)
	if fpc[0]!=0:
		tpc=np.r_[0, tpc]
		fpc = np.r_[0, fpc]
	# If one of the classes was empty, return NaN
	if fpc[-1] == 0 or tpc[-1] == 0:
		return np.nan
	# Convert sums to rates
	tpr = tpc / tpc[-1]
	fpr = fpc / fpc[-1]
	# Calculate area under the curve using trapezoidal rule
	auc = np.trapz(tpr, fpr, axis=0)
	if want_curve:
		curve = np.hstack([fpr.reshape((-1,1)),tpr.reshape((-1,1))])
		return auc,curve
	return auc
def boostrap_auc(z, y, ntrial =10):
	n = len(y)
	aucs = []
	for t in range(ntrial):
		sample = np.random.randint(0,n,n)
		zt = z[sample].copy().reshape((-1,1))
		yt = y[sample].copy().reshape((-1,1))
		auc = calc_auc(zt,yt)
		if np.isnan(auc):
			return np.nan,np.nan
		aucs.append(auc)
	return np.mean(aucs), np.std(aucs)
def calc_metrics(z, y, aucthresh=(0.5,0.5)):
	metric={}
	z = np.array(z)
	y=np.array(y)
	M = ~np.isnan(y)
	z=z[M]
	y=y[M]
	metric['pearson.r'], metric['pearson.p'] = scipy.stats.pearsonr(z,y)
	metric["spearman.r"], metric["spearman.p"] = scipy.stats.spearmanr(y,z)
	#print(y)
	#print(z)
	if np.any(y< aucthresh[0]) and np.any(y>aucthresh[1]):
		lo,hi = aucthresh
		Mlo = y < lo
		Mhi = y >= hi
		y[Mlo] = 0
		y[Mhi] =1
		M = np.logical_or(Mlo, Mhi)
		y=y[M]
		z = z[M]
		metric["auc"] = calc_auc(z, y)
		#metric["auc.mean"], metric["auc.std"] = boostrap_auc(z, y, 10)
	return metric




def printmatrix(a,index):
	print("Printing matrix")
	for i in range(len(a)):
		if index[i]:
			print(a[i,:])

def printmaskarr(mask, arr1, arr2):
	l1 =[]
	l2=[]
	for i in range(len(mask)):
		if mask[i] == 1.0:
			l1.append(arr1[i])
			l2.append(arr2[i])
	print("arr1:",l1)
	print("arr2",l2)

def accuracy(mask, list1, list2):
	correct = np.equal(list1, list2)
	
	acc_all = np.array(correct, dtype=np.float32)

	mask = mask.flatten()
	mask /= np.mean(mask)
	acc_all*= mask
	return np.mean(acc_all)

def one_task(seqs, adjs_0, adjs_1, idx_train, idx_val, idx_test, idx_all, labels, options, save=False):
	n = len(labels)
	start_time = time.time()
	test_mask = sample_mask(idx_test, n)
	val_mask = sample_mask(idx_val, n)
	train_mask = sample_mask(idx_train, n)
	all_mask = sample_mask(idx_all, n)
	#print(test_mask)
	#print(val_mask)
	#print(train_mask)

	y_train = np.zeros(labels.shape)
	y_val = np.zeros(labels.shape)
	y_test = np.zeros(labels.shape)

	y_train[train_mask,:] = labels[train_mask,:]
	y_test[test_mask,:] = labels[test_mask,:]
	y_val[val_mask,:] = labels[val_mask,:]
	test_size = len(idx_test)
	val_size = len(idx_val)
	train_size = len(idx_train)

	psize = adjs_0[0].shape[0] - n

	# concatenate
	padding_mask = np.zeros(psize)
	padding_y = np.zeros((psize, 1))

	train_mask = np.concatenate([train_mask, padding_mask])
	test_mask = np.concatenate([test_mask, padding_mask])
	val_mask = np.concatenate([val_mask, padding_mask])
	all_mask = np.concatenate([all_mask, padding_mask])

	train_mask=np.reshape(train_mask, [len(train_mask),1])
	val_mask=np.reshape(val_mask, [len(train_mask),1])
	test_mask=np.reshape(test_mask, [len(train_mask),1])
	all_mask=np.reshape(all_mask, [len(train_mask),1])

	y_train = np.concatenate((y_train, padding_y), axis=0)
	y_val = np.concatenate((y_val, padding_y), axis=0)
	y_test = np.concatenate((y_test, padding_y), axis=0)

	print(adjs_0[0].shape)
	print(adjs_1[0].shape)


	#define placeholders
	if options['model'] == 'gcn':
		support_0 = [preprocess_adj(adj) for adj in adjs_0]
		support_1 = [preprocess_adj(adj) for adj in adjs_1]
		num_support = len(adjs_0)
	features_0 = scipy.sparse.identity(adjs_0[0].shape[0])
	features_0 = preprocess_features(features_0)
	features_1 = scipy.sparse.identity(adjs_1[0].shape[0])
	features_1 = preprocess_features(features_1)

	tsize = adjs_0[0].shape[0]
	placeholders = {
		'support_0':[tf.sparse_placeholder(tf.float32) for _ in range(num_support)],
		'support_1':[tf.sparse_placeholder(tf.float32) for _ in range(num_support)],
		#'features':tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
		'features_0':tf.sparse_placeholder(tf.float32, shape=None),
		'features_1':tf.sparse_placeholder(tf.float32, shape=None),
		'labels':tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
		'labels_mask': tf.placeholder(tf.int32),
		'dropout': tf.placeholder_with_default(0., shape=()),
		'training': tf.placeholder_with_default(0., shape=())
	}
	
	#build the model
	model = dGCN("gcn", placeholders, features_0[2][1], options)
	#Initializing session
	sess = tf.Session()
	
	# define model evaluation function
	def evaluate(features_0, features_1, support_0, support_1, labels, mask, placeholders):
		feed_dict_val = construct_feed_dict(
			features_0, features_1, support_0, support_1, labels, mask, placeholders)
		outs_val = sess.run([model.predloss, model.accuracy, model.preds, model.labels, model.debug], feed_dict=feed_dict_val)
		return outs_val[0], outs_val[1], outs_val[2], outs_val[3], outs_val[4]

	sess.run(tf.global_variables_initializer())
	cost_val=[]
	# train model
	feed_dict = construct_feed_dict(features_0, features_1, support_0, support_1, y_train, train_mask, placeholders)
	feed_dict.update({placeholders['dropout']:0.0})
	valcost, valacc, preds, labels, debug = evaluate(features_0, features_1, support_0, support_1, y_val, val_mask, placeholders)
	#print("epoch %d, val loss = %f,  valid acc = %f" %(-1, valcost, valacc))

	for epoch in range(options['epochs']):
		feed_dict = construct_feed_dict(features_0, features_1, support_0, support_1, y_train, train_mask, placeholders)
		feed_dict.update({placeholders['dropout']:options['dropout']})
		#if epoch != 0:
		outs = sess.run([model.opt_op, model.loss, model.accuracy, model.debug], feed_dict=feed_dict)
		
		#printmatrix(outs[-1], train_mask)
		

		
		#print("acc=%f" % accuracy(train_mask, pred, label))

		if epoch % 10 == 0:
			feed_dict = construct_feed_dict(features_0, features_1, support_0, support_1, y_train, train_mask, placeholders)
			feed_dict.update({placeholders['dropout']:0.0})
			valloss, valacc, preds, labels, debug = evaluate(features_0, features_1, support_0, support_1, y_val, val_mask, placeholders)
			#valloss = np.sqrt(valloss)
			#valacc = np.sqrt(valacc)

			val_pred = []
			val_labels = []
			for i in range(len(val_mask)):
				if val_mask[i] == 1.0:
					val_pred.append(sigmoid(preds[i]))
					val_labels.append(labels[i][0])
			#for y,z in zip(val_labels, val_pred):
			#	print(y,z)
			#print(val_pred)
			auc=metrics.roc_auc_score(val_labels, val_pred)
			print("epoch %d: trn lost = %f, valid loss = %f, auc = %f" %(epoch, outs[1], valloss, auc))
			print(" running in %s second" % (time.time()-start_time))
			#print("test loss = %f, auc_test = %f" % (t_loss, auc_test))
			#print("pcc = %f, scc = %f" % (pcc, scc))
			

			#testloss, testacc, preds, labels, debug = evaluate(features, support, y_test, test_mask, placeholders)

		#if epoch > 100 and cost_val[-1] >= np.mean(cost_val[-10:-1]):
		#	print("Early stopping ....")
		#	break
		#if epoch % 100 ==0:
#
		if epoch > 100:
			if save:

				word_inv_vocab_0 = construct_inv_vocab(seqs, SIZE[0])
				'''
				word_inv_vocab_1 = construct_inv_vocab(seqs, SIZE[1])
				len1= len(word_inv_vocab_1)
				total = len(word_inv_vocab_0) + len(word_inv_vocab_1)
				for key in word_inv_vocab_0.keys():
					value = word_inv_vocab_0[key]
					keynew = key + len1
					if keynew in word_inv_vocab_1.keys():
						print("DKM")
						exit(1)
					word_inv_vocab_1[keynew] = value
				#print(word_inv_vocab_1)
				#print(len(word_inv_vocab_0), len(word_inv_vocab_1), total)
				'''
				[reps, tlabels, masks, scores] = outs[3]
				masks = all_mask
				print(all_mask)
				print(len(all_mask))
				print(np.sum(all_mask))
				tlabels = labels
				
				masks = masks.tolist()
				scores = scores.tolist()
				_scores = []
				count = 0
				for i, mask in enumerate(masks):
					if mask[0] == 0:
						_scores.append(scores[i][0])
						
					else:
						count +=1
						#print(scores[i][0], tlabels[i])



				#print(all_mask)
				#seqs = discovermotif(_scores, word_inv_vocab_0, 5)
				#print(seqs)
				#prefix = 'vis3_'
				#file_reps = prefix + "reps"
				#file_labels = prefix + "labels"
				#file_masks = prefix + "masks"

				#np.save(file_reps, reps)
				#np.save(file_labels, tlabels)
				#np.save(file_masks, masks)
			break
	# test
	#print("Opt finished")
	#testing
	#feed_dict = construct_feed_dict(features, support, y_train, train_mask,placeholders)
	#feed_dict.update({placeholders['dropout']:0.0})
	#outs = evaluate(features, support, y_test, test_mask, placeholders)
	#t_loss = np.sqrt(outs[0])
	#t_acc = np.sqrt(outs[1])
	#print("test loss: %f, test acc: %f" % (t_loss, t_acc))
	#preds = outs[2]
	#labels = outs[3]
	
	#p_labels = preds
	#test_pred = []
	#test_labels = []
	#for i in range(len(test_mask)):
	#	if test_mask[i] == 1.0:
	#		test_pred.append(p_labels[i])
	#		test_labels.append(labels[i])

	#print(test_pred)
	#print(test_labels)
	#auc = calc_auc(np.array(test_labels), np.array(test_pred))
	#pcc,_ = scipy.stats.pearsonr(test_pred, test_labels)
	#scc,_ = scipy.stats.spearmanr(test_pred, test_labels)
	#print("Acc = %f, Pcc: %f, scc: %f"% (t_acc, pcc, scc))
	#test_pred = np.array(test_pred)
	#test_labels =np.array(test_labels)
	
	#auc = calc_auc(test_pred, test_labels)

	#print("****auc = %f"% (auc))

	#return auc
	

#if len(sys.argv) != 2:
#	dataset = ''
#else:
dataset = sys.argv[1]

start_time = time.time()

# Settings
options = {}
options['balance'] = 1.0
options['model'] = 'gcn'
options['epochs'] = 5000
options['dropout'] = 0.5
options['weight_decay'] = 0.00001
options['hidden1'] = 400
options['hidden2'] = 100
options['learning_rate'] = 0.008#0.0001 for small ds #0.02 for hitsflip#0.02 for pbm

if dataset == 'hitsflip':
	seqs = load_data_HITSFLIP(dataset)
	n = len(seqs)
	n_fold = 10
	tag = label_folds(n, n_fold)

if dataset == 'encode':
	
	
	#tfids=["CHD2_H1-hESC_CHD2_(AB68301)_Stanford"]#,"CHD2_H1-hESC_CHD2_(AB68301)_Stanford",
	#"CHD2_HeLa-S3_CHD2_(AB68301)_Stanford","CHD2_HepG2_CHD2_(AB68301)_Stanford",
	#"CHD2_K562_CHD2_(AB68301)_Stanford"]#,
	#tfids=["ARID3A_HepG2_ARID3A_(NB100-279)_Stanford","ARID3A_K562_ARID3A_(sc-8821)_Stanford",
	#"ATF3_H1-hESC_ATF3_HudsonAlpha","ATF3_HepG2_ATF3_HudsonAlpha","ATF3_K562_ATF3_HudsonAlpha"]
	#"MAX_HUVEC_Max_Stanford","MAX_HeLa-S3_Max_Stanford","MAX_HepG2_Max_Stanford",
	#"MAX_K562_Max_HudsonAlpha","MAX_K562_Max_Stanford","MAX_NB4_Max_Stanford"]
	#generateAdjs(tfids)
	#exit(1)
	

	#tfid=tfids[0]s
	tfid = "ATF3_K562_ATF3_HudsonAlpha"
	local_weights = [0.0, 0.1,0.5,1.0]
	train_seqs = load_encode_train(tfid)
	test_seqs = load_encode_test(tfid)
	if len(test_seqs) <= 550:
		print("generating background seqs for testing ds")
		test_seqs=generate_encode(tfid)
	rev_train_seqs = load_encode_train(tfid, use_reverse=True)
	rev_test_seqs = load_encode_test(tfid, use_reverse=True)
	#print(test_seqs)
	#test_seqs = train_seqs
	print("number of training seques: %d" % len(train_seqs))
	print("number of testing seques: %d" % len(test_seqs))
	print("number of reversed training seques: %d" % len(rev_train_seqs))
	print("number of reversed testing seques: %d" % len(rev_test_seqs))

	seqs = train_seqs + test_seqs
	rev_seqs = rev_train_seqs + rev_test_seqs

	n = len(seqs)
	n_train = len(train_seqs)
	n_test = len(test_seqs)
	n_fold = 3
	tag = label_folds(n, n_fold)
	for i in range(n_test):
		tag[n_train + i] = n_fold + 1
	#print(tag)
	#print(tag[-100:-1])

print(options)
print("Infor about the dataset:")
print("No of sequences: %d, length: %d" % (len(seqs), len(seqs[0][0])))
#np.random.shuffle(seqs)

labels = [seq[1] for seq in seqs]
#labels = np.array(labels)
#m = np.mean(labels)
#labels_t = [label - m for label in labels]
labels_t = []
for label in labels:
	if label == 0.0:
		labels_t.append(0.0)
	else:
		labels_t.append(1.0)

labels = np.reshape(np.array(labels_t), [n, 1])

# construct adjacent matrices
sizes = [8,10]
ext = ".npz"

strsize =""
for size in sizes:
	strsize+=str(size)
	
filename = "./adjs/"+tfid + "_" + strsize + "_" + "adj3" + ext
rev_filename = "./adjs/"+tfid + "_" + strsize + "_" + "_rev_adj3" + ext
if os.path.exists(filename):
	adj3 = scipy.sparse.load_npz(filename)
	rev_adj3 = scipy.sparse.load_npz(rev_filename)
	print("load " + filename)
else:
	adj3, rev_adj3 = rev_d2w_global(seqs, rev_seqs, sizes)
	#adj3 = d2w_global(seqs, sizes)
	scipy.sparse.save_npz(filename, adj3)
	scipy.sparse.save_npz(rev_filename, rev_adj3)
	print("save " + filename)

filename = "./adjs/"+tfid + "_" + strsize + "_" + "adj2" + ext
rev_filename = "./adjs/"+tfid + "_" + strsize + "_" + "_rev_adj2" + ext
if os.path.exists(filename):
	adj2 = scipy.sparse.load_npz(filename)
	rev_adj2 = scipy.sparse.load_npz(rev_filename)
	print("load " + filename)
else:
	#adj2 = w2w_local(seqs, sizes, 1.0)
	adj2, rev_adj2 = rev_w2w_local(seqs, rev_seqs, sizes, 1.0)
	scipy.sparse.save_npz(filename, adj2)
	scipy.sparse.save_npz(rev_filename, rev_adj2)
	print("save " + filename)

filename = "./adjs/"+tfid + "_" + strsize + "_" + "adj1" + ext
rev_filename = "./adjs/"+tfid + "_" + strsize + "_" + "_rev_adj1" + ext
if os.path.exists(filename):
	adj1 = scipy.sparse.load_npz(filename)
	rev_adj1 = scipy.sparse.load_npz(rev_filename)
	print("load " + filename)
else:
	#adj1 = w2w_global(seqs, sizes)
	adj1, rev_adj1=rev_w2w_global(seqs, rev_seqs, sizes)
	scipy.sparse.save_npz(filename, adj1)
	scipy.sparse.save_npz(rev_filename, rev_adj1)
	print("save " + filename)

aucs =[]
for local_w in local_weights:
	print("local w = %f" %local_w)
	adj1 = 5.0 * adj1# 5.0
	rev_adj1 = 5.0 * rev_adj1#5.0
	adj2 = local_w * adj2
	rev_adj2 = local_w * rev_adj2
	#print(adj1.shape)
	adjs_0=[adj3+adj1+adj2]
	adjs_1=[adj3+adj1+adj2]
	idx_test = np.where(tag == n_fold+1)
	idx_val = np.where(tag == n_fold+1)
	idx_all = np.arange(len(tag))
	tmp = np.setdiff1d(idx_all, idx_test)
	idx_train = np.setdiff1d(tmp, idx_val)
	time_start1 = time.time()
	metric = one_task(seqs, adjs_0, adjs_1, idx_train, idx_val, idx_test, idx_all, labels, options, True)
	#print("the inner process running in %s second" % (time.time()-time_start1))
#print(" running in %s second" % (time.time()-start_time))







