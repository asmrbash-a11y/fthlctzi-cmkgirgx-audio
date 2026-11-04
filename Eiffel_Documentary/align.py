#!/usr/bin/env python3
"""Align the user's Arabic transcript to the supplied narration using word timings.

Input is local Whisper word timing JSONL and the supplied script. Exact text comes
from the user, not the speech recognizer. All times are in seconds from audio start.
"""
import json, re, unicodedata
from pathlib import Path
from rapidfuzz import fuzz
from PIL import ImageFont

P=Path(__file__).resolve().parent
DURATION=332.72

def norm(s):
    s=unicodedata.normalize('NFKC',s)
    s=re.sub('[\u064b-\u065f\u0670\u0640]', '', s)
    s=s.translate(str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ٱ':'ا','ى':'ي','ة':'ه','ؤ':'و','ئ':'ي','ک':'ك'}))
    return re.sub(r'[^\u0621-\u064a0-9]', '', s)

paragraphs=[p.strip() for p in (P/'script_ar.txt').read_text(encoding='utf8').split('\n\n') if p.strip()]
target=[]
for pno,p in enumerate(paragraphs):
    for text in p.split():
        target.append({'word':text,'norm':norm(text),'paragraph':pno})
raw=[]
for line in (P/'asr_recognition.jsonl').read_text(encoding='utf8').splitlines():
    d=json.loads(line)
    for w in d['words']:
        parts=w['word'].strip().split()
        if not parts: continue
        start,end=float(w['start']),float(w['end'])
        weights=[max(1,len(norm(x))) for x in parts]; total=sum(weights); cursor=start
        for i,(part,weight) in enumerate(zip(parts,weights)):
            xend=end if i==len(parts)-1 else cursor+(end-start)*weight/total
            n=norm(part)
            if n:raw.append({'word':part,'norm':n,'start':cursor,'end':xend,'probability':w.get('probability',0)})
            cursor=xend

n,m=len(target),len(raw)
gap=0.60
# Full dynamic programming provides a monotonic global alignment; byte backpointers.
prev=[j*gap for j in range(m+1)]
backs=[bytearray(m+1) for _ in range(n+1)]
for j in range(1,m+1):backs[0][j]=2
for i in range(1,n+1):
    cur=[i*gap]+[0.0]*m; backs[i][0]=1
    x=target[i-1]['norm']
    for j in range(1,m+1):
        y=raw[j-1]['norm']
        sim=1.0 if x==y else fuzz.ratio(x,y)/100
        if min(len(x),len(y))<=2 and x!=y: sim*=0.70
        sub=(1-sim)*1.15
        diag=prev[j-1]+sub; delete=prev[j]+gap; insert=cur[j-1]+gap
        if diag<=delete and diag<=insert:cur[j]=diag;backs[i][j]=0
        elif delete<=insert:cur[j]=delete;backs[i][j]=1
        else:cur[j]=insert;backs[i][j]=2
    prev=cur
pairs={};i=n;j=m
while i or j:
    op=backs[i][j]
    if i and j and op==0:
        pairs[i-1]=j-1;i-=1;j-=1
    elif i and (op==1 or not j):i-=1
    else:j-=1

unmapped=[];low=[]
for i,t in enumerate(target):
    if i in pairs:
        a=raw[pairs[i]]
        t['start'],t['end']=a['start'],a['end']
        t['heard']=a['word']
        t['similarity']=round(fuzz.ratio(t['norm'],a['norm'])/100,3)
        if t['similarity']<0.5:low.append((i,t['word'],a['word'],t['start']))
    else:unmapped.append(i)

# Allocate unmatched words within the timing gaps bounded by neighboring recognized words.
# When the recognizer fused adjacent words, carve the span evenly from the most
# similar neighboring word instead of leaving the unmatched word invisible.
i=0
while i<n:
    if i in pairs:i+=1;continue
    j=i
    while j+1<n and j+1 not in pairs:j+=1
    prevword=target[i-1] if i else None
    nextword=target[j+1] if j+1<n else None
    a=prevword['end'] if prevword else 0.0
    b=nextword['start'] if nextword else DURATION
    count=j-i+1
    if b-a < 0.13*count:
        # A fused ASR token: borrow a little of the surrounding phonetic span.
        if prevword and prevword['end']-prevword['start']>0.35:
            borrow=min((prevword['end']-prevword['start'])*0.45,0.22*count)
            prevword['end']-=borrow;a=prevword['end']
        if b-a < 0.13*count and nextword and nextword['end']-nextword['start']>0.35:
            borrow=min((nextword['end']-nextword['start'])*0.45,0.22*count)
            nextword['start']+=borrow;b=nextword['start']
    if b<=a:b=a+0.10*count
    lens=[max(1,len(target[k]['norm'])) for k in range(i,j+1)]
    total=sum(lens);cursor=a
    for k,w in zip(range(i,j+1),lens):
        end=cursor+(b-a)*w/total
        target[k].update(start=round(cursor,3),end=round(end,3),heard='',similarity=0)
        cursor=end
    i=j+1

# Numbers are frequently normalized to a single digit token while the ASR also
# emits the final spoken number words afterwards; absorb those into the number.
for i,t in enumerate(target[:-1]):
    if t['norm'].isdigit() and i in pairs and i+1 in pairs:
        k1,k2=pairs[i],pairs[i+1]
        if k2>k1+1:
            extras=raw[k1+1:k2]
            if all(re.fullmatch(r'[\u0621-\u064a]+',x['norm']) for x in extras):
                if extras[-1]['end']-t['end']<2.0:t['end']=extras[-1]['end']

for k,t in enumerate(target):
    t.pop('norm',None)
    t['start']=round(max(0,t['start']),3);t['end']=round(min(DURATION,max(t['end'],t['start']+0.02)),3)
    if k and t['start'] < target[k-1]['start']:
        t['start']=target[k-1]['start']

(P/'words.json').write_text(json.dumps(target,ensure_ascii=False,indent=2),encoding='utf8')
para=[]
for pno,p in enumerate(paragraphs):
    ws=[w for w in target if w['paragraph']==pno]
    para.append({'index':pno,'start':ws[0]['start'],'end':ws[-1]['end'],'text':p,'words':len(ws)})
(P/'paragraphs.json').write_text(json.dumps(para,ensure_ascii=False,indent=2),encoding='utf8')
print(f'Words {n}; recognized tokens {m}; matched {len(pairs)}; interpolated {len(unmapped)}; low confidence {len(low)}')
print('--- interpolated ---')
for k in unmapped:print(k,target[k]['word'],target[k]['start'],target[k]['end'])
print('--- uncertain matches ---')
for x in low:print(x)
print('--- paragraphs ---')
for p in para:print(f"{p['index']:02d} {p['start']:6.2f}-{p['end']:6.2f} {p['text'][:72]}")
