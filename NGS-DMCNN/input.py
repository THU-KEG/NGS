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

# whole dataset
in_path_train = "./data/data_train/"
in_path_test = "./data/data_test/"
in_path_develop = "./data/data_develop/"
out_path = "./data/data_DMCNN/"
wordVec_path = './data/wordVec_glove_6b/'

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

word_vec_mat = np.load(wordVec_path + 'wordVec.npy')
word2id = json.load(open(wordVec_path + 'word2id.json', "r"))
id2word = {v: k for k, v in word2id.items()}

def gen_wordEmb(sen):
	wordEmb = np.zeros((SenLen), dtype = np.int64)
	words = nltk.word_tokenize(sen)
	for k, word in enumerate(words):
		if k < SenLen:
			if word in word2id:
				wordEmb[k] = word2id[word]
			else:
				wordEmb[k] = word2id['UNK']
	for k in range(k + 1, SenLen):
		wordEmb[k] = word2id['BLANK']
	return wordEmb

class Dataset:
	def __init__(self, Tag):
		self.tag = Tag
		self.Tword = []
		self.Tsubtype = []
		self.Tlocal = []
		self.Tpos = []
		self.TmaskR = []
		self.TmaskL = []
		self.TsenLabel = []
		self.Tfilename = []

		self.Aword = []
		self.Asubtype = []
		self.Arole = []
		self.Ain_sen = []
		self.Alength = []
		self.AsenLabel = []
		self.AmaskR = []
		self.AmaskL = []
		self.AmaskM = []
		self.Apos1 = []
		self.Apos2 = []
		self.Aloc = []
		self.Aloc_mark = []
		self.Aarg_role = []

		self.file_num = 0
		self.Tins_num = 0
		self.Ains_num = 0
		self.event_num = 1
		self.ins_num_tri = 0
		self.sen_num = 0
		self.sen_idx = 0
		self.sen_pos = []
		self.sen_dic = {}
		self.ett_dic = {}
		self.ins_has_ett = {}

	def find_senID(self, word_begin, word_end):
		for ID in self.sen_dic:
			if word_begin in self.sen_dic[ID][1] and word_end in self.sen_dic[ID][1]:
				return ID

	def get_senRange(self, text, para_begin, para_end):
		cur_text = nltk.sent_tokenize(text)
		for i, sen in enumerate(cur_text):
			idx = text.find(sen)
			sen_range = range(idx + para_begin, idx + para_begin + len(sen))
			etts = []
			trigger = []
			self.sen_dic.update({self.sen_idx: [sen, sen_range, etts, trigger]})
			self.sen_idx += 1
			self.ins_num_tri += len(nltk.word_tokenize(sen))

	def gen_Tins(self, word, length, tri_idx, word_idx, subtype, ID):
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
		self.TmaskR.append(maskR)
		self.TmaskL.append(maskL)
		self.Tpos.append(pos)
		self.Tlocal.append(local)
		self.Tsubtype.append(label)
		self.TsenLabel.append(ID)
		self.Tfilename.append(self.filename + ' ')

		return maskR, maskL, pos, local, label

	def gen_Ains(self, word, tri_idx, ett_idx, length, ett_length, ID):
		maskR = np.zeros((SenLen), dtype = np.float32)
		maskL = np.zeros((SenLen), dtype = np.float32)
		maskM = np.zeros((SenLen), dtype = np.float32)
		pos1 = np.zeros((SenLen), dtype = np.int64)	
		pos2 = np.zeros((SenLen), dtype = np.int64)
		# subtype = np.zeros((SenLen), dtype = np.int64)
		loc = []
		index_min = min(tri_idx, ett_idx + ett_length - 1)
		index_max = max(tri_idx, ett_idx + ett_length - 1)

		for j in range(SenLen):
			if j < length:
				pos1[j] = j - min(tri_idx, SenLen) + SenLen
				# pos2[j] = j - min(ett_idx, SenLen) + SenLen
				if j < ett_idx:
					pos2[j] = j - min(ett_idx, SenLen) + SenLen
				elif j > ett_idx + ett_length -1:
					pos2[j] = j - min(ett_idx + ett_length - 1, SenLen) + SenLen
				else:
					pos2[j] = SenLen
			else:
				pos1[j] = 0
				pos2[j] = 0

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
		
		if tri_idx > SenLen - 1:
			loc.extend([0,0,0])
		else:
			loc.extend([word[max(0, tri_idx - 1)], word[tri_idx], word[min(tri_idx + 1, SenLen - 1)]])
		if ett_idx > SenLen - 1:
			loc.append(0)
			for i in range(ett_idx, ett_idx + ett_length):
				loc.append(0)
			loc.append(0)
		else:
			loc.append(word[ett_idx - 1])
			for i in range(ett_idx, ett_idx + ett_length):
				loc.append(word[min(i, SenLen - 1)])
			loc.append(word[min(SenLen - 1, ett_idx + ett_length)])
		for i in range(len(loc), SenLen):
			loc.append(0)
		loc_mark = ett_length

		# generate the argRoles
		arg_role = np.zeros((SenLen), dtype = np.int64)
		for ett in self.ins_has_ett:
			arg_in_sen = list(range(self.ins_has_ett[ett][1], min(self.ins_has_ett[ett][1] + self.ins_has_ett[ett][2], SenLen)))
			if ett != ID:
				for i in arg_in_sen:
					if arg_role[i] == 0 or arg_role[i] == Ett_tag:
						arg_role[i] = self.ins_has_ett[ett][0]
		arg_in_sen = list(range(self.ins_has_ett[ID][1], min(self.ins_has_ett[ID][1] + self.ins_has_ett[ID][2], SenLen)))
		arg_role[arg_in_sen] = predEtt_tag
		if tri_idx < SenLen:
			arg_role[tri_idx] = tri_tag

		self.AmaskR.append(maskR)
		self.AmaskL.append(maskL)
		self.AmaskM.append(maskM)
		self.Apos1.append(pos1)
		self.Apos2.append(pos2)
		self.Aloc.append(loc)
		self.Aloc_mark.append(loc_mark)
		self.Aarg_role.append(arg_role)
		return maskR, maskL, maskM, pos1, pos2, loc, loc_mark, arg_role

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
				# print (self.filename, ett_ID, ett, ett_begin, ett_end, sen_ID)
				pass

	def Dataloader(self, in_path, out_path):
		path_list = os.listdir(in_path)

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

				# process the .sgm files
				f = open(text_file_name,"r")
				lines = f.readlines()
				para_begin = 0
				para_end = 0
				text = ""
				flag = 0
				first = 1
				self.sen_idx=0
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

				# process the .apf files
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
				flag = self.Tri_process_pos(ev_mention)
				if flag == False:
					# print (flag)
					continue
				else:
					ev_arguments = ev_mention.getElementsByTagName("event_mention_argument")
					self.ins_has_ett = {}
					self.Find_args(ev_arguments)
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
		length = len(nltk.word_tokenize(text))
		if length > SenLen:
			return False
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
			
		text_before_anchor = text[:tar_begin]
		if text_before_anchor.find('&') != -1:
			if text_before_anchor[text_before_anchor.find('&'):text_before_anchor.find('&')+5] != '&amp;' and self.filename not in wrong_file:
				tar_begin -= 4
				text_before_anchor = text[:tar_begin]
		tri_idx = len(nltk.word_tokenize(text_before_anchor))
		word = gen_wordEmb(text)
		words = nltk.word_tokenize(text)
		if trigger_word == "war" and words[tri_idx - 1] == "postwar":
			tri_idx -= 1
		maskR, maskL, pos, local, subtype = self.gen_Tins(word, len(words), tri_idx + len(nltk.word_tokenize(trigger_word)) - 1, tri_idx, self.subtype, self.sen_num+sen_ID)
		self.sen_dic[sen_ID][3].append(trigger_word.split()[0])
		
		self.sen_begin = sen_begin
		self.sen_end = sen_end
		self.text = text
		self.sen_ID = sen_ID
		self.tri_idx = tri_idx
		self.sen_pos.append(sen_ID)
		return True

	def Tri_process_neg(self, sen_ID):
		text = self.sen_dic[sen_ID][0]
		words = nltk.word_tokenize(text)
		if len(words) > SenLen:
			return False
		word = gen_wordEmb(text)
		for word_idx, i in enumerate(words):
			if i not in self.sen_dic[sen_ID][3]:
				maskR, maskL, pos, local, subtype = self.gen_Tins(word, len(words), 0, word_idx, 0, sen_ID + self.sen_num)

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
				ett_in_sen = len(nltk.word_tokenize(text_before_ett))
				ett_length = len(nltk.word_tokenize(self.ett_dic[ID][0]))
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
					ett_in_sen = len(nltk.word_tokenize(text_before_ett))
					ett_length = len(nltk.word_tokenize(self.ett_dic[ID][0]))
					if nltk.word_tokenize(self.text)[ett_in_sen - 1] == 'lets' and self.sen_dic[self.sen_ID][0] == 's':
						ett_in_sen -= 1
					role = Ett_tag
					self.ins_has_ett.update({ID: [role, ett_in_sen, ett_length]})

	def Arg_process(self, ev_arguments):
		word = gen_wordEmb(self.text)
		for ett in self.ins_has_ett:
			self.Aword.append(word)
			self.Asubtype.append(self.subtype)
			if self.ins_has_ett[ett][0] == Ett_tag:
				self.Arole.append(0)
			else:
				self.Arole.append(self.ins_has_ett[ett][0])
			self.Ain_sen.append(self.ins_has_ett[ett][1])
			self.Alength.append(self.ins_has_ett[ett][2])
			self.AsenLabel.append(self.sen_ID+self.sen_num)
			maskR, maskL, maskM, pos1, pos2, loc, loc_mark, arg_role = self.gen_Ains(word, self.tri_idx, self.ins_has_ett[ett][1], len(nltk.word_tokenize(self.text)), self.ins_has_ett[ett][2], ett)
			
	def file_save(self, Tag, out_path):
		self.Tins_num = len(self.Tword)
		self.Ains_num = len(self.Aword)
		print (Tag, '| file_num:', self.file_num)
		print (Tag, '| sen_num:', self.sen_num)
		print (Tag, '| word_num:', self.ins_num_tri)
		print (Tag, '| Tins_num:', self.Tins_num)
		print (Tag, '| Ains_num:', self.Ains_num)
		print (Tag, '| Ains_num_pos:', len(np.nonzero(self.Arole)[0]))

		print ("\n***")
		print ("Trigger instances:")
		print (Tag, '| Tins_num:', self.Tins_num)
		print (Tag, '| trigger_num:', self.event_num)
		
		Tori_sen = []
		for idx in range(self.Tins_num):
			ori_sen = ''
			for i in range(len(self.Tword[idx])):
				if id2word[self.Tword[idx][i]] != 'BLANK':
					ori_sen += id2word[self.Tword[idx][i]] + ' '
			Tori_sen.append(ori_sen)
		idx = 0
		print (Tag, '| ins_idx:', idx)
		print (Tag, '| Tword[idx]:', self.Tword[idx])
		
		print (Tag, '| ori_sen[idx]:', Tori_sen[idx])
		print (Tag, '| Tsubtype[idx]:', self.Tsubtype[idx])
		print (Tag, '| TmaskL[idx]:', self.TmaskL[idx])
		print (Tag, '| TmaskR[idx]:', self.TmaskR[idx])
		print (Tag, '| Tpos[idx]:', self.Tpos[idx])
		print (Tag, '| Tlocal[idx]:', self.Tlocal[idx])
		print (Tag, '| TsenLabel[idx]:', self.TsenLabel[idx])
		print (Tag, '| Tfilename[idx]:', self.Tfilename[idx])

		print ("\n***")
		print ("Argument instances:")
		print (Tag, '| Ains_num:', self.Ains_num)

		Aori_sen = []
		for idx in range(self.Ains_num):
			ori_sen = ''
			for i in range(len(self.Aword[idx])):
				if id2word[self.Aword[idx][i]] != 'BLANK':
					ori_sen += id2word[self.Aword[idx][i]] + ' '
			Aori_sen.append(ori_sen)

		idx = 5
		print (Tag, '| idx:', idx)
		print (Tag, '| Aword[idx]:', self.Aword[idx])
		print (Tag, '| ori_sen[idx]:', Aori_sen[idx])
		print (Tag, '| Arole[idx]:', self.Arole[idx])
		print (Tag, '| Asubtype[idx]:', self.Asubtype[idx])
		print (Tag, '| Aarg_role[idx]:', self.Aarg_role[idx])
		print (Tag, '| AsenLabel[idx]:', self.AsenLabel[idx])	# 记录该entity属于哪个句子，在Gibbs sampling时使用
		print (Tag, '| Aloc[idx]:', self.Aloc[idx])
		print (Tag, '| Aloc_mark[idx]:', self.Aloc_mark[idx])
		print (Tag, '| Apos1[idx]:', self.Apos1[idx])
		print (Tag, '| Apos2[idx]:', self.Apos2[idx])
		print (Tag, '| AmaskL[idx]:', self.AmaskL[idx])
		print (Tag, '| AmaskM[idx]:', self.AmaskM[idx])
		print (Tag, '| AmaskR[idx]:', self.AmaskR[idx])
		print (Tag, '| Ain_sen[idx]:', self.Ain_sen[idx])
		print (Tag, '| Alength[idx]:', self.Alength[idx])

		print ("***")
		print ("Saving files...")
		np.save(os.path.join(out_path, Tag + '_wordEmb.npy'), self.Tword)
		np.save(os.path.join(out_path, Tag + '_label.npy'), self.Tsubtype)
		np.save(os.path.join(out_path, Tag + '_posEmb.npy'), self.Tpos)
		np.save(os.path.join(out_path, Tag + '_local.npy'), self.Tlocal)
		np.save(os.path.join(out_path, Tag + '_maskL.npy'), self.TmaskL)
		np.save(os.path.join(out_path, Tag + '_maskR.npy'), self.TmaskR)
		np.save(os.path.join(out_path, Tag + '_senLabel.npy'), self.TsenLabel)
		if Tag == 'test':
			with open(out_path + Tag + '_filename', 'w') as fp:
				fp.writelines(self.Tfilename)

		np.save(os.path.join(out_path, Tag + '_wordEmb_arg.npy'), self.Aword)
		np.save(os.path.join(out_path, Tag + '_pos1Emb_arg.npy'), self.Apos1)
		np.save(os.path.join(out_path, Tag + '_pos2Emb_arg.npy'), self.Apos2)
		np.save(os.path.join(out_path, Tag + '_local_arg.npy'), self.Aloc)
		np.save(os.path.join(out_path, Tag + '_localMark_arg.npy'), self.Aloc_mark)
		np.save(os.path.join(out_path, Tag + '_subtype_arg.npy'), self.Asubtype)
		np.save(os.path.join(out_path, Tag + '_argRole_arg.npy'), self.Aarg_role)
		np.save(os.path.join(out_path, Tag + '_Senlabel_arg.npy'), self.AsenLabel)
		np.save(os.path.join(out_path, Tag + '_maskR_arg.npy'), self.AmaskR)
		np.save(os.path.join(out_path, Tag + '_maskL_arg.npy'), self.AmaskL)
		np.save(os.path.join(out_path, Tag + '_maskM_arg.npy'), self.AmaskM)
		np.save(os.path.join(out_path, Tag + '_label_arg.npy'), self.Arole)
		np.save(os.path.join(out_path, Tag + '_ettIdx_arg.npy'), self.Ain_sen)
		np.save(os.path.join(out_path, Tag + '_ettLength_arg.npy'), self.Alength)
		np.save(os.path.join(out_path, 'wordVec.npy'), word_vec_mat)
		word2id_json = json.dumps(word2id)
		with open(out_path + 'word2id.json', 'w') as json_file:
			json_file.write(word2id_json)

		print ("Finish saving.")

if __name__=='__main__':
	trainSet = Dataset('train')
	developSet = Dataset('develop')
	testSet = Dataset('test')
	trainSet.Dataloader(in_path_train, out_path)
	developSet.Dataloader(in_path_develop, out_path)
	testSet.Dataloader(in_path_test, out_path)