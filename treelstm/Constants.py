PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'



'''
aux : aux , auxpass, cop
comp : acomp, ccomp, xcomp,  pcomp
obj : dobj, iobj, pobj
subj : nsubj, nsubjpass, csubj , csubjpass
adv : neg, advcl, npadvmod, tmod, advmod
poss : poss,  possesive
'''

merged_relations = {
'aux' : 'aux',
'auxpass' : 'aux',
'cop' : 'aux',

'acomp' : 'comp' ,
'ccomp' : 'comp',
'xcomp' : 'comp',
'pcomp': 'comp',

'dobj' : 'obj',
'iobj' : 'obj',
'pobj' : 'obj',

'nsubj' : 'subj',
'nsubjpass' : 'subj',
'csubj' : 'subj',
'csubjpass' : 'subj',

'neg' : 'adv',
'tmod' : 'adv',
'advcl' : 'adv',
'npadvmod': 'adv',
'rcmod' : 'adv',
'vmod' :'adv,',
'advmod': 'adv',
'amod' : 'adv',

# 'advmod': 'mod',
# 'amod': 'mod,'

'prep':'other',
'det' : 'other',
}


