import pickle
import random
import sys
import os

def load(fn,default=[]):
    if os.path.isfile(fn):
        with open(fn, 'rb') as f:
            return pickle.load(f)
    return default

if not os.path.isdir('res'):
    os.mkdir('res')
    
dialogues = load('res/dialogues.p', {})
your_sents = load('res/your_sents.p')
responder_lines = load('res/other_sents.p')

chat_file = sys.argv[1]
SPEAKER_NAME = sys.argv[2]
RESPONDER_NAME = sys.argv[3]

with open(chat_file,'r', encoding="utf8") as f:
    content = f.readlines()

prev = None
for line in content[1:]:
    if True in [x in line for x in [
        'Session Start',
        'Session Close',
        'Auto-response',
        'You have been disconnected',
        'signed on at',
        'signed off at',
        'is trying to send you ',   
        '*** You have received'
    ]]:
        continue
        
        
    if '%s|'%SPEAKER_NAME in line:
        text = line.split('%s|'%SPEAKER_NAME)[-1]
        your_sents.append(text)
        if prev == 'None':
            continue
        if prev == 'pr':
            dialogues[responder_lines[-1]] = text
        prev = 'sp'
    elif '%s|'%RESPONDER_NAME in line:
        text = line.split('%s|'%RESPONDER_NAME)[-1]
        responder_lines.append(text)
        prev = 'pr'
    else:
        print('Who said this? ' + line)        
        if prev == 'sp':
            your_sents[-1] += line
        elif prev == 'pr':
            responder_lines[-1] += line


f = open('res/dialogues.p','wb')
pickle.dump(dialogues,f)
f.close()

f = open('res/your_sents.p','wb')
pickle.dump(your_sents,f)
f.close()

f = open('res/other_sents.p','wb')
pickle.dump(responder_lines,f)
f.close()
