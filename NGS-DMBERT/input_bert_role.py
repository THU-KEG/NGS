import os
import json
import xml.dom.minidom
import nltk
import nltk.data
import numpy as np
import pickle
import sys
import copy
from constant import * 
from pytorch_pretrained_bert import BertTokenizer
from string import punctuation
import re

in_path_train = "./data/data_train/"
in_path_test = "./data/data_test/"
in_path_develop = "./data/data_develop/"
out_path = "./data/data_DMBERT/"

if not os.path.exists(out_path):
	os.mkdir(out_path)

EventSub2id = dict(
	[('NONE_type',0),('Be-Born',1),('Die',2),('Marry',3),
	('Divorce',4),('Injure',5),('Transfer-Ownership',6),
	('Transfer-Money',7),('Transport',8),('Start-Org',9),
	('End-Org',10),('Declare-Bankruptcy',11),('Merge-Org',12),
	('Attack',13),('Demonstrate',14),('Meet',15),('Phone-Write',16),
	('Start-Position',17),('End-Position',18),('Nominate',19),
	('Elect',20),('Arrest-Jail',21),('Release-Parole',22),
	('Charge-Indict',23),('Trial-Hearing',24),('Sue',25),
	('Convict',26),('Sentence',27),('Fine',28),('Execute',29),
	('Extradite',30),('Acquit',31),('Pardon',32),('Appeal',33)
	])

Role2id = dict(
	[('NONE_role',0),('Person',1),('Place',2),('Buyer',3),
	('Seller',4),('Beneficiary',5),('Price',6),('Artifact',7),
	('Origin',8),('Destination',9),('Giver',10),('Recipient',11),
	('Money',12),('Org',13),('Agent',14),('Victim',15),
	('Instrument',16),('Entity',17),('Attacker',18),('Target',19),
	('Defendant',20),('Adjudicator',21),('Prosecutor',22),('Plaintiff',23),
	('Crime',24),('Position',25),('Sentence',26),('Vehicle',27),
	('Time-Within',28),('Time-Starting',29),('Time-Ending',30),
	('Time-Before',31),('Time-After',32),('Time-Holds',33),
	('Time-At-Beginning',34),('Time-At-End',35)
	])
id2Role = {v: k for k, v in Role2id.items()}

wrong_file = []

punc = punctuation + '.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|:：'
# print (punc)

def Print_file(file_name, item_name, item):
    with open('results/' + file_name, 'a') as f:
        f.write(item_name + ':\n')
        f.write(str(item) + '\n')

class Dataset:
	def __init__(self, Tag):
		self.tag = Tag
		self.Tword = []
		self.Tin_mask = []
		self.Tsubtype = []
		self.Tlocal = []
		self.Tpos = []
		self.TmaskR = []
		self.TmaskL = []
		self.TsenLabel = []
		self.Tfilename = []

		self.Aword = []
		self.Ain_mask = []
		self.Asubtype = []
		self.Arole = []
		self.Ain_sen = []
		self.Alength = []
		self.Atri_idx = []
		self.Atri_length = []
		self.AsenLabel = []
		self.AmaskR = []
		self.AmaskL = []
		self.AmaskM = []
		self.Ainsert_tags = []
		self.Aword_prob = []
		self.Ain_mask_prob = []
		self.AmaskM_prob = []
		self.AmaskR_prob = []
		self.AmaskL_prob = []
		self.AmaskType_prob = []
		self.Ain_sen_prob = []
		self.Alength_prob = []
		self.Atri_idx_prob = []
		self.Ainsert_tags_prob = []

		self.file_num = 0
		self.Tins_num = 0
		self.Ains_num = 0
		self.event_num = 0
		self.ins_num_tri = 0
		self.sen_num = 0
		self.sen_pos = []
		self.sen_dic = {}
		self.ett_dic = {}
		self.ins_has_ett = {}
		self.tokenizer = BertTokenizer.from_pretrained("../BERT_CACHE/bert-base-uncased-vocab.txt")
		assert self.tokenizer != None

	def find_senID(self, word_begin, word_end):
		for ID in self.sen_dic:
			if word_begin in self.sen_dic[ID][1] and word_end in self.sen_dic[ID][1]:
				return ID

	def get_senRange(self, text, para_begin, para_end):
		cur_text = nltk.sent_tokenize(text)	# 分句
		for i, sen in enumerate(cur_text):
			idx = text.find(sen)
			sen_range = range(idx + para_begin, idx + para_begin + len(sen))
			etts = []
			trigger = []
			self.sen_dic.update({self.sen_idx: [sen, sen_range, etts, trigger]})
			self.sen_idx += 1
			self.ins_num_tri += len(self.tokenizer.tokenize(sen))

	def gen_Tins(self, word, input_mask, length, tri_idx, word_idx, subtype, ID):
		maskR = np.zeros((SenLen), dtype = np.float32)
		maskL = np.zeros((SenLen), dtype = np.float32)
		pos = np.zeros((SenLen), dtype = np.int64)
		local = np.zeros(3, dtype = np.int64)
		for j in range(SenLen):
			if j < length and word_idx < SenLen:
				pos[j] = j - word_idx + SenLen
			else:
				pos[j] = 0
			if pos[j] < 0:
				print ('Error! j:{} , pos[j]:{} , word_idx:{}'.format(j, pos[j], word_idx))
				raise ValueError
			if j >= length:
				maskR[j] = 0
				maskL[j] = 0
			elif j - word_idx <= 0:
				maskR[j] = 0
				maskL[j] = 1
			else:
				maskR[j] = 1
				maskL[j] = 0
		if word_idx == 0:
			local[0] = 0
			local[1] = word[0]
			local[2] = word[1]
		elif word_idx == SenLen - 1:
			local[0] = word[SenLen - 2]
			local[1] = word[SenLen - 1]
			local[2] = 0
		elif word_idx == SenLen:
			local[0] = word[SenLen - 1]
			local[1] = 0
			local[2] = 0
		elif word_idx > SenLen:
			local[0] = 0
			local[1] = 0
			local[2] = 0
		else:
			local[0] = word[word_idx - 1]
			local[1] = word[word_idx]
			local[2] = word[word_idx + 1]
		if word_idx == tri_idx:
			label = subtype
		else:
			label = 0

		self.Tword.append(word)
		self.Tin_mask.append(input_mask)
		self.TmaskR.append(maskR)
		self.TmaskL.append(maskL)
		self.Tsubtype.append(label)
		self.Tfilename.append(self.filename + ' ')

		return maskR, maskL, pos, local, label

	def gen_Ains(self, word, tri_idx, ett_idx, ett_length, ID):
		# insert tokens
		length = len(word)
		maskR = np.zeros((SenLen), dtype = np.float32)
		maskL = np.zeros((SenLen), dtype = np.float32)
		maskM = np.zeros((SenLen), dtype = np.float32)
		index_min = min(tri_idx, ett_idx + ett_length - 1)
		index_max = max(tri_idx, ett_idx + ett_length - 1)

		for j in range(SenLen):
			if j >= length:
				maskL[j] = 0
				maskM[j] = 0
				maskR[j] = 0
			elif j - index_min <= 0:
				maskL[j] = 1
				maskM[j] = 0
				maskR[j] = 0
			elif j - index_max <= 0:
				maskL[j] = 0
				maskM[j] = 1
				maskR[j] = 0
			else:
				maskL[j] = 0
				maskM[j] = 0
				maskR[j] = 1

		input_mask = [1]*len(word)
		oriLen = len(word)
		if len(word) > SenLen:
			word = word[:SenLen]
			input_mask = input_mask[:SenLen]
		else:
			L = len(word)
			for i in range(0, SenLen - L):
				word.append(0)
				input_mask.append(0)

		return word, input_mask, maskR, maskL, maskM

	def gen_ettDic(self, entity_mentions):
		for ett_mention in entity_mentions:
			ett_ID = ett_mention.getAttribute("ID")
			pos_ID = ett_ID.find("-")
			ett_ID = ett_ID[pos_ID + 1: ]
			ett_charseq = ett_mention.getElementsByTagName("extent")[0].getElementsByTagName("charseq")[0]
			ett = ett_charseq.childNodes[0].data.lower().replace('\n',' ')
			ett_begin = int(ett_charseq.getAttribute("START"))
			ett_end = int(ett_charseq.getAttribute("END"))
			sen_ID = self.find_senID(ett_begin, ett_end)
			if sen_ID != None and self.sen_dic[sen_ID][0].find(ett) != -1:
				self.ett_dic.update( {ett_ID: [ett, ett_begin, ett_end, sen_ID]} )
				self.sen_dic[sen_ID][2].append(ett_ID)
			else:
				pass

	def Dataloader(self, in_path, out_path):
		path_list = os.listdir(in_path)
		# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

		print ("---------------")
		print("Loading data...")
		for filename in path_list:
			# process a file
			if os.path.splitext(filename)[1] == '.sgm':
				self.filename = filename
				sen_idx = 0
				self.sen_dic = {}
				self.ett_dic = {}
				text_file_name = in_path + filename
				apf_file_name = in_path + os.path.splitext(filename)[0] + '.apf.xml'

				# process the sgm files
				f = open(text_file_name, "r", encoding = 'utf-8')
				lines = f.readlines()
				para_begin = 0
				para_end = 0
				text = ""
				flag = 0
				first = 1
				self.sen_idx = 0
				for line_idx, line in enumerate(lines):
					if line == '<TEXT>\n':
						flag = 1
					elif line == '</TEXT>\n':
						flag = 0

					if (line[0] == ' ' or line[0] == '<' or line[0] == '\n') and text != '':
						self.get_senRange(text, para_begin, para_end)
						text = ''
						para_begin = para_end

					para_end_pre = copy.deepcopy(para_end)
					para_end += len(line)
					if line.find('&') != -1:
						if line[line.find('&'):line.find('&')+5] != '&amp;' and self.filename not in wrong_file:
							para_end += 4
					if line[0] == '<' and line[1] == '/':
						para_end -= (len(line) - 1)
					elif line[0] == '<' and line[1] != '/':
						i1 = 0
						i2 = line.find('>')
						i3 = line.find('</')
						i4 = line.find('>',i2 + 1)
						# print (i1, i2, i3, i4)
						if i1 != -1 and i2 != -1:
							para_end -= (i2 - i1 + 1)
						if i3 != -1 and i4 != -1:
							para_end -= (i4 - i3 + 1)
					elif line[0:2] == '  ':
						para_end -= len(line)
						if line[4] == '<':
							para_end += 1
						elif line[4] == '"' and first == 1:
							para_end += 2
							first = 0
						elif line[4] == '"':
							para_end += 4
					elif flag == 1 and line != '\n':
						if text == '':
							para_begin = para_end_pre
						text_new = (line[:-1] + ' ').lower()
						text = text + text_new
				
				# for sen_id in self.sen_dic:
				# 	print (self.sen_dic[sen_id])
				# sys.exit()

				# processing the apf files
				self.apf = xml.dom.minidom.parse(apf_file_name)
				self.root = self.apf.documentElement
				self.events = self.root.getElementsByTagName("event")
				entities = self.root.getElementsByTagName("entity")
				timex2s = self.root.getElementsByTagName("timex2")
				values = self.root.getElementsByTagName("value")
				for entity in entities:
					entity_mentions = entity.getElementsByTagName("entity_mention")
					self.gen_ettDic(entity_mentions)
				for timex2 in timex2s:
					timex2_mentions = timex2.getElementsByTagName("timex2_mention")
					self.gen_ettDic(timex2_mentions)
				for value in values:
					value_mentions = value.getElementsByTagName("value_mention")
					self.gen_ettDic(value_mentions)

				# for sen_id in self.sen_dic:
				# 	Print_file ('Aresult', str(sen_id), self.sen_dic[sen_id])

				self.sen_pos = []
				self.process_one_file()
				self.file_num += 1
				if self.file_num % 50 == 0:
					print (self.file_num)
				self.sen_num += len(self.sen_dic)

		self.file_save(self.tag, out_path)	

	def process_one_file(self):
		# print (self.filename)
		for event in self.events:
			self.subtype = EventSub2id[event.getAttribute("SUBTYPE")]
			self.event_mentions = event.getElementsByTagName("event_mention")
			for ev_mention in self.event_mentions:
				self.ins_has_ett = {}
				flag = self.Tri_process_pos(ev_mention)
				if flag == False:
					# print (flag)
					continue
				else:
					ev_arguments = ev_mention.getElementsByTagName("event_mention_argument")
					self.Find_args(ev_arguments)
					# print (self.ins_has_ett)
					self.Arg_process(ev_arguments)
					self.event_num += 1
		for sen_ID in self.sen_dic:
			self.Tri_process_neg(sen_ID)

	def Tri_process_pos(self, ev_mention):
		anchor = ev_mention.getElementsByTagName("anchor")[0]
		trigger = anchor.getElementsByTagName("charseq")[0]
		ldc_scope = ev_mention.getElementsByTagName("ldc_scope")[0]
		sen = ldc_scope.getElementsByTagName("charseq")[0]
		text = sen.childNodes[0].data.replace('\n',' ').lower()

		trigger_word = trigger.childNodes[0].data.lower().replace('\n',' ')	# for processing the event_argument
		length = len(self.tokenizer.tokenize(text))
		sen_begin = int(sen.getAttribute("START"))
		sen_end = int(sen.getAttribute("END"))
		tar_begin = int(trigger.getAttribute("START")) - sen_begin
		for sen_ID in self.sen_dic:
			if int(trigger.getAttribute("START")) in self.sen_dic[sen_ID][1]:
				break
		if self.sen_dic[sen_ID][0].find(trigger_word) == -1:
			for i in self.sen_dic:
				if self.sen_dic[i][0] in text or text in self.sen_dic[i][0]:
					sen_ID = i
		if self.sen_dic[sen_ID][0].find(trigger_word) == -1:
			return False

		if sen_begin != list(self.sen_dic[sen_ID][1])[0]:
			if (self.sen_dic[sen_ID][0][0] == '"' and list(self.sen_dic[sen_ID][1])[0] == sen_begin - 1) or (self.sen_dic[sen_ID][0][0] == '`' and list(self.sen_dic[sen_ID][1])[0] == sen_begin - 2):
				pass
			elif self.sen_dic[sen_ID][0].find(trigger_word) == -1:
				print (self.filename)
				print (int(trigger.getAttribute("START")), trigger_word)
				print ('text:', text)
				print ('sen:', self.sen_dic[sen_ID][0])
				print (sen_begin, self.sen_dic[sen_ID][1])
				print ('\n')
				# for sen_id in self.sen_dic:
				# 	print (self.sen_dic[sen_id])
			
		text_before_anchor = text[:tar_begin]
		if text_before_anchor.find('&') != -1:
			if text_before_anchor[text_before_anchor.find('&'):text_before_anchor.find('&')+5] != '&amp;' and self.filename not in wrong_file:
				tar_begin -= 4
				text_before_anchor = text[:tar_begin]
		tri_idx = len(self.tokenizer.tokenize(text_before_anchor)) + 1
		tokens = self.tokenizer.tokenize(text)
		words = ["[CLS]"] + tokens + ["[SEP]"]
		if trigger_word == "war" and words[tri_idx - 1] == "postwar":
			tri_idx -= 1
		self.tri_idx = copy.deepcopy(tri_idx)
		self.tri_length = len(self.tokenizer.tokenize(trigger_word))

		word = self.tokenizer.convert_tokens_to_ids(words)
		input_mask = [1]*len(word)
		oriLen = len(word)
		if len(word) > SenLen:
			# return False
			word = word[:SenLen]
			words = words[:SenLen]
			input_mask = input_mask[:SenLen]
		else:
			L = len(word)
			for i in range(0, SenLen - L):
				word.append(0)
				input_mask.append(0)

		maskR, maskL, pos, local, subtype = self.gen_Tins(word, input_mask, len(words), tri_idx + len(self.tokenizer.tokenize(trigger_word)) - 1, tri_idx, self.subtype, sen_ID)
		self.TsenLabel.append(self.sen_num+sen_ID)
		self.sen_dic[sen_ID][3].append(self.tokenizer.tokenize(trigger_word)[0])
		self.sen_begin = sen_begin
		self.sen_end = sen_end
		self.text = text
		self.sen_ID = sen_ID
		self.sen_pos.append(sen_ID)
		return True

	def Tri_process_neg(self, sen_ID):
		text = self.sen_dic[sen_ID][0]
		tokens = self.tokenizer.tokenize(text)
		words = ["[CLS]"] + tokens + ["[SEP]"]
		word = self.tokenizer.convert_tokens_to_ids(words)
		input_mask = [1]*len(word)
		oriLen = len(word)
		if len(word) > SenLen:
			# return False
			word = word[:SenLen]
			words = words[:SenLen]
			input_mask = input_mask[:SenLen]
		else:
			L = len(word)
			for i in range(0, SenLen - L):
				word.append(0)
				input_mask.append(0)

		for word_idx, i in enumerate(words):
			if i not in self.sen_dic[sen_ID][3]:
				maskR, maskL, pos, local, subtype = self.gen_Tins(word, input_mask, len(words), 0, word_idx, 0, sen_ID)
				self.TsenLabel.append(self.sen_num + sen_ID)

	def Find_args(self, ev_arguments):
		for ev_argument in ev_arguments:
			REFID = ev_argument.getAttribute("REFID")
			role = Role2id[ev_argument.getAttribute("ROLE")]
			pos_ID = REFID.find("-")
			ID = REFID[pos_ID + 1:]
			if self.ett_dic.__contains__(ID):
				ett_begin = self.ett_dic[ID][1] - self.sen_begin
				text_before_ett = self.text[:ett_begin]
				if text_before_ett.find('&') != -1:
					if text_before_ett[text_before_ett.find('&'):text_before_ett.find('&')+5] != '&amp;' and self.filename not in wrong_file:
						ett_begin -= 4
						text_before_ett = self.text[:ett_begin]
				ett_in_sen = len(self.tokenizer.tokenize(text_before_ett))
				ett_length = len(self.tokenizer.tokenize(self.ett_dic[ID][0]))
				self.ins_has_ett.update({ID: [role, ett_in_sen, ett_length]})

		for ID in self.sen_dic[self.sen_ID][2]:
			if not self.ins_has_ett.__contains__(ID):
				if self.ett_dic[ID][1] >= self.sen_begin and self.ett_dic[ID][2] <= self.sen_end and ID[0] != 'T' and ID[0] != 'V':
					ett_begin = self.ett_dic[ID][1] - self.sen_begin
					text_before_ett = self.text[:ett_begin]
					if text_before_ett.find('&') != -1:
						if text_before_ett[text_before_ett.find('&'):text_before_ett.find('&')+5] != '&amp;' and self.filename not in wrong_file:
							ett_begin -= 4
							text_before_ett = self.text[:ett_begin]
					ett_in_sen = len(self.tokenizer.tokenize(text_before_ett))
					ett_length = len(self.tokenizer.tokenize(self.ett_dic[ID][0]))
					if self.tokenizer.tokenize(self.text)[ett_in_sen - 1] == 'lets' and self.sen_dic[self.sen_ID][0] == 's':
						ett_in_sen -= 1
					role = Ett_tag
					self.ins_has_ett.update({ID: [role, ett_in_sen, ett_length]})

		# Print_file('Aresult', 'ins_has_ett', self.ins_has_ett)

	def Arg_process(self, ev_arguments):
		tokens = self.tokenizer.tokenize(self.text)
		purewords = ["[CLS]"] + tokens + ["[SEP]"]

		for ett in self.ins_has_ett:
			if isinstance(ett, str) == True:
				words = copy.deepcopy(purewords)
				ett_length = self.ins_has_ett[ett][2]
				ett_idx = self.ins_has_ett[ett][1] + 1
				tri_idx = self.tri_idx
				tri_length = self.tri_length
				insert_token = "[unused" + str(self.subtype + 37) + "]"
				insert_tags = np.ones_like(words).tolist()
				words.insert(tri_idx + tri_length, insert_token)
				insert_tags.insert(tri_idx + tri_length, 0)
				words.insert(tri_idx, insert_token)
				insert_tags.insert(tri_idx, 0)
				insert_tags[0] = 0
				insert_tags[-1] = 0
				# print (words)
				# print (insert_tags)
				assert len(words) == len(insert_tags)
				# sys.exit()
				if ett_idx >= tri_idx:
					ett_idx += 2
				tri_idx += 1
				tri_length += 1
				word = self.tokenizer.convert_tokens_to_ids(words)
				word, input_mask, maskR, maskL, maskM = self.gen_Ains(word, tri_idx + tri_length - 1, ett_idx, ett_length, ett)
				self.Aword.append(word)
				self.Ain_mask.append(input_mask)
				self.AmaskR.append(maskR)
				self.AmaskL.append(maskL)
				self.AmaskM.append(maskM)
				self.Ainsert_tags.append(insert_tags)
				self.Ain_sen.append(ett_idx)
				self.Asubtype.append(self.subtype)
				self.Alength.append(ett_length)
				self.Atri_idx.append(tri_idx)
				self.Atri_length.append(tri_length)

		for ett in self.ins_has_ett:
			if isinstance(ett, str) == True:
				tri_idx = self.tri_idx
				tri_length = self.tri_length
				ett_length = self.ins_has_ett[ett][2]
				ett_idx = self.ins_has_ett[ett][1] + 1
				words = copy.deepcopy(purewords)
				role_list = []
				idx_list = []
				for ID in self.ins_has_ett:
					if ID != ett:
						role_list.append(self.ins_has_ett[ID][0])
						role_list.append(self.ins_has_ett[ID][0])
					else:
						role_list.append(Ett_tag)
						role_list.append(Ett_tag)
					idx_list.append(self.ins_has_ett[ID][1] + 1)
					idx_list.append(self.ins_has_ett[ID][1] + self.ins_has_ett[ID][2] + 1)

				data=[(idx, role) for idx, role in zip(idx_list,role_list)]
				data.sort(reverse = True)
				idx_list=[idx for idx,role in data]
				role_list=[role for idx,role in data]
				# print (idx_list)
				# print (role_list)
				for i, idx in enumerate(idx_list):
					insert_token = "[unused" + str(role_list[i]) + "]"
					words.insert(idx, insert_token)
					if ett_idx >= idx:
						ett_idx += 1
					elif ett_idx < idx and ett_idx + ett_length >= idx:
						ett_length += 1
					if tri_idx >= idx:
						tri_idx += 1
					elif tri_idx < idx and tri_idx + tri_length >= idx:
						tri_length += 1
				insert_token = "[unused" + str(self.subtype + 37) + "]"
				words.insert(len(words), insert_token)

				word = self.tokenizer.convert_tokens_to_ids(words)
				word_prob, input_mask_prob, maskR_prob, maskL_prob, maskM_prob = self.gen_Ains(word, tri_idx + tri_length - 1, ett_idx, ett_length, ett)
				maskType_prob = np.zeros((SenLen), dtype = np.float32)
				self.Aword_prob.append(word_prob)
				self.Ain_mask_prob.append(input_mask_prob)
				self.AmaskR_prob.append(maskR_prob)
				self.AmaskL_prob.append(maskL_prob)
				self.AmaskM_prob.append(maskM_prob)
				self.AmaskType_prob.append(maskType_prob)
				self.Ain_sen_prob.append(ett_idx)
				self.Alength_prob.append(ett_length)
				self.Atri_idx_prob.append(tri_idx)

				if self.ins_has_ett[ett][0] == Ett_tag:
					self.Arole.append(0)
				else:
					self.Arole.append(self.ins_has_ett[ett][0])
				self.AsenLabel.append(self.sen_ID+self.sen_num)	
			
	def file_save(self, Tag, out_path):
		print ('Saving...')
		
		idx = 0
		while idx < len(self.Aword_prob):
			if self.Aword_prob[idx][-1] != 0 and self.Aword_prob[idx][-1] != 102:	#[PAD] and [SEP]
				# print (idx, self.Aword_prob[idx][-1])
				self.Aword = np.delete(self.Aword, idx, axis = 0)
				self.Ain_mask = np.delete(self.Ain_mask, idx, axis = 0)
				self.Arole = np.delete(self.Arole, idx, axis = 0)
				self.AsenLabel = np.delete(self.AsenLabel, idx, axis = 0)
				self.Asubtype = np.delete(self.Asubtype, idx, axis = 0)
				self.AmaskL = np.delete(self.AmaskL, idx, axis = 0)
				self.AmaskM = np.delete(self.AmaskM, idx, axis = 0)
				self.AmaskR = np.delete(self.AmaskR, idx, axis = 0)
				self.Ain_sen = np.delete(self.Ain_sen, idx, axis = 0)
				self.Alength = np.delete(self.Alength, idx, axis = 0)
				self.Atri_idx = np.delete(self.Atri_idx, idx, axis = 0)
				self.Atri_length = np.delete(self.Atri_length, idx, axis = 0)

				self.Aword_prob = np.delete(self.Aword_prob, idx, axis = 0)
				self.Ain_mask_prob = np.delete(self.Ain_mask_prob, idx, axis = 0)
				self.AmaskM_prob = np.delete(self.AmaskM_prob, idx, axis = 0)
				self.AmaskL_prob = np.delete(self.AmaskL_prob, idx, axis = 0)
				self.AmaskR_prob = np.delete(self.AmaskR_prob, idx, axis = 0)
				self.Ain_sen_prob = np.delete(self.Ain_sen_prob, idx, axis = 0)
				self.Alength_prob = np.delete(self.Alength_prob, idx, axis = 0)
				self.Atri_idx_prob = np.delete(self.Atri_idx_prob, idx, axis = 0)
				idx -= 1
			idx += 1
		
		self.Tins_num = len(self.Tword)
		self.Ains_num = len(self.Aword)
		print (Tag, '| file_num:', self.file_num)
		print (Tag, '| sen_num:', self.sen_num)
		print (Tag, '| word_num:', self.ins_num_tri)
		print (Tag, '| Tins_num:', self.Tins_num)
		print (Tag, '| Ains_num:', self.Ains_num, len(self.Aword_prob), len(self.Asubtype))
		print (Tag, '| Ains_num_pos:', len(np.nonzero(self.Arole)[0]))

		print ("\n***")
		print ("Trigger instances:")
		print (Tag, '| Tins_num:', self.Tins_num)
		print (Tag, '| trigger_num:', self.event_num)
		
		Tori_sen = []
		for word in self.Tword:
			ori_sen = self.tokenizer.convert_ids_to_tokens(word)
			Tori_sen.append(ori_sen)
		idx = 4
		print (Tag, '| ins_idx:', idx)
		print (Tag, '| Tword[idx]:', self.Tword[idx])
		print (Tag, '| Tin_mask[idx]:', self.Tin_mask[idx])
		print (Tag, '| ori_sen[idx]:', Tori_sen[idx])
		print (Tag, '| Tsubtype[idx]:', self.Tsubtype[idx])
		print (Tag, '| TmaskL[idx]:', self.TmaskL[idx])
		print (Tag, '| TmaskR[idx]:', self.TmaskR[idx])
		# print (Tag, '| Tpos[idx]:', self.Tpos[idx])
		# print (Tag, '| Tlocal[idx]:', self.Tlocal[idx])
		print (Tag, '| TsenLabel[idx]:', self.TsenLabel[idx])
		print (Tag, '| Tfilename[idx]:', self.Tfilename[idx])

		print ("\n***")
		print ("Argument instances:")
		print (Tag, '| Ains_num:', self.Ains_num)

		Aori_sen = []
		for word in self.Aword:
			ori_sen = self.tokenizer.convert_ids_to_tokens(word)
			Aori_sen.append(ori_sen)

		for idx in range(len(self.Aword)):
			if self.Ain_mask[idx][-1] == 1:
				break
		# idx = 3
		print (Tag, '| idx:', idx)
		print (Tag, '| ori_sen[idx]:', Aori_sen[idx])
		print (Tag, '| Aword[idx]:', self.Aword[idx])
		print (Tag, '| Ain_mask[idx]:', self.Ain_mask[idx])
		print (Tag, '| Arole[idx]:', self.Arole[idx])
		print (Tag, '| AsenLabel[idx]:', self.AsenLabel[idx])
		print (Tag, '| Asubtype[idx]:', self.Asubtype[idx])
		print (Tag, '| AmaskL[idx]:', self.AmaskL[idx])
		print (Tag, '| AmaskM[idx]:', self.AmaskM[idx])
		print (Tag, '| AmaskR[idx]:', self.AmaskR[idx])
		print (Tag, '| Ain_sen[idx]:', self.Ain_sen[idx])
		print (Tag, '| Alength[idx]:', self.Alength[idx])
		print (Tag, '| Atri_idx[idx]:', self.Atri_idx[idx])
		print (Tag, '| Atri_length[idx]:', self.Atri_length[idx])

		Aori_sen_prob = []
		for word in self.Aword_prob:
			ori_sen = self.tokenizer.convert_ids_to_tokens(word)
			Aori_sen_prob.append(ori_sen)
		print (Tag, '| ori_sen[idx]:', Aori_sen_prob[idx])
		print (Tag, '| Aword_prob[idx]:', self.Aword_prob[idx])
		print (Tag, '| Ain_mask_prob[idx]:', self.Ain_mask_prob[idx])
		print (Tag, '| AmaskL_prob[idx]:', self.AmaskL_prob[idx])
		print (Tag, '| AmaskM_prob[idx]:', self.AmaskM_prob[idx])
		print (Tag, '| AmaskR_prob[idx]:', self.AmaskR_prob[idx])
		print (Tag, '| AmaskType_prob[idx]:', self.AmaskType_prob[idx])
		print (Tag, '| Ain_sen_prob[idx]:', self.Ain_sen_prob[idx])
		print (Tag, '| Alength_prob[idx]:', self.Alength_prob[idx])
		print (Tag, '| Atri_idx_prob[idx]:', self.Atri_idx_prob[idx])
		# print ('self.TsenLabel:', self.TsenLabel)
		# print ('self.AsenLabel', self.AsenLabel)

		print ("***")
		print ("Saving files...")
		# data for trigger
		np.save(os.path.join(out_path, Tag + '_wordEmb.npy'), self.Tword)
		np.save(os.path.join(out_path, Tag + '_inMask.npy'), self.Tin_mask)
		np.save(os.path.join(out_path, Tag + '_label.npy'), self.Tsubtype)
		# np.save(os.path.join(out_path, Tag + '_posEmb.npy'), self.Tpos)
		# np.save(os.path.join(out_path, Tag + '_local.npy'), self.Tlocal)
		np.save(os.path.join(out_path, Tag + '_maskL.npy'), self.TmaskL)
		np.save(os.path.join(out_path, Tag + '_maskR.npy'), self.TmaskR)
		np.save(os.path.join(out_path, Tag + '_senLabel.npy'), self.TsenLabel)
		if Tag == 'test':
			with open(out_path + Tag + '_filename', 'w') as fp:
				fp.writelines(self.Tfilename)

		# data for argument
		np.save(os.path.join(out_path, Tag + '_wordEmb_arg.npy'), self.Aword)
		np.save(os.path.join(out_path, Tag + '_inMask_arg.npy'), self.Ain_mask)
		np.save(os.path.join(out_path, Tag + '_maskR_arg.npy'), self.AmaskR)
		np.save(os.path.join(out_path, Tag + '_maskL_arg.npy'), self.AmaskL)
		np.save(os.path.join(out_path, Tag + '_maskM_arg.npy'), self.AmaskM)
		np.save(os.path.join(out_path, Tag + '_label_arg.npy'), self.Arole)
		np.save(os.path.join(out_path, Tag + '_subtype_arg.npy'), self.Asubtype)
		np.save(os.path.join(out_path, Tag + '_ettIdx_arg.npy'), self.Ain_sen)
		np.save(os.path.join(out_path, Tag + '_ettLength_arg.npy'), self.Alength)
		np.save(os.path.join(out_path, Tag + '_senLabel_arg.npy'), self.AsenLabel)
		np.save(os.path.join(out_path, Tag + '_triIdx_arg.npy'), self.Atri_idx)
		np.save(os.path.join(out_path, Tag + '_triLength_arg.npy'), self.Atri_length)

		np.save(os.path.join(out_path, Tag + '_wordEmb_prob.npy'), self.Aword_prob)
		np.save(os.path.join(out_path, Tag + '_inMask_prob.npy'), self.Ain_mask_prob)
		np.save(os.path.join(out_path, Tag + '_maskR_prob.npy'), self.AmaskR_prob)
		np.save(os.path.join(out_path, Tag + '_maskL_prob.npy'), self.AmaskL_prob)
		np.save(os.path.join(out_path, Tag + '_maskM_prob.npy'), self.AmaskM_prob)
		np.save(os.path.join(out_path, Tag + '_maskType_prob.npy'), self.AmaskType_prob)
		np.save(os.path.join(out_path, Tag + '_ettIdx_prob.npy'), self.Ain_sen_prob)
		np.save(os.path.join(out_path, Tag + '_ettLength_prob.npy'), self.Alength_prob)
		np.save(os.path.join(out_path, Tag + '_triIdx_prob.npy'), self.Atri_idx_prob)

		print ("Finish saving.")

if __name__=='__main__':
	trainSet = Dataset('train')
	developSet = Dataset('develop')
	testSet = Dataset('test')
	trainSet.Dataloader(in_path_train, out_path)
	developSet.Dataloader(in_path_develop, out_path)
	testSet.Dataloader(in_path_test, out_path)

