#!/usr/bin/env python3
"""Build the vertical Arabic documentary from local audio, images and word timings.

Requirements: Python 3, numpy, opencv-python, Pillow with libraqm, scipy, and ffmpeg
with libx264/AAC. No external images, music, fonts or calls are needed at render time.
"""
import argparse, bisect, json, math, os, subprocess, sys, time, wave
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

P=Path(__file__).resolve().parent
W,H,FPS=576,1024,24
DURATION=332.72
WORDS=json.loads((P/'words.json').read_text(encoding='utf8'))
PARAS=json.loads((P/'paragraphs.json').read_text(encoding='utf8'))
AS=P/'assets'
FONTS=P/'fonts'
cv2.setNumThreads(1)

# Consistent visual identity, all derived from theme.json.
theme=json.loads((P/'theme.json').read_text(encoding='utf8'))
IVORY=(243,234,213); GOLD=(223,178,99); RED=(151,44,53); INK=(24,28,29)
FBLACK=str(FONTS/'Tajawal-Black.ttf'); FBOLD=str(FONTS/'Tajawal-Bold.ttf'); FREG=str(FONTS/'Tajawal-Regular.ttf'); FINTER=str(FONTS/'Inter-VF.ttf')
def font(size,which='black'):
    return ImageFont.truetype({'black':FBLACK,'bold':FBOLD,'regular':FREG,'inter':FINTER}[which],size=size,layout_engine=ImageFont.Layout.RAQM)
def arabic(draw,xy,text,f,fill,anchor='mm',**kwargs):
    draw.text(xy,text,font=f,fill=fill,direction='rtl',language='ar',anchor=anchor,**kwargs)
def centered(draw,xy,text,f,fill):
    draw.text(xy,text,font=f,fill=fill,anchor='mm')
def smooth(x):
    x=max(0,min(1,x));return x*x*(3-2*x)

def rgba_to_cv(im):
    a=np.asarray(im.convert('RGBA'),dtype=np.uint8)
    return np.ascontiguousarray(a[:,:,[2,1,0,3]])

def shadowed(canvas,card,xy,rad=8,opacity=120):
    shadow=Image.new('RGBA',canvas.size,(0,0,0,0))
    alpha=card.getchannel('A').point(lambda a:int(a*opacity/255))
    shadow.paste((0,0,0,opacity),xy,alpha)
    shadow=shadow.filter(ImageFilter.GaussianBlur(rad))
    canvas.alpha_composite(shadow)
    canvas.alpha_composite(card,xy)

def title_card():
    im=Image.new('RGBA',(W,240),(0,0,0,0));d=ImageDraw.Draw(im)
    d.rounded_rectangle((20,10,W-20,220),radius=4,fill=(14,18,20,224),outline=(205,171,109,200),width=1)
    d.rectangle((35,25,242,60),fill=(133,39,49,255))
    arabic(d,(138,42),'وثائقي للمعرفة',font(26,'bold'),IVORY)
    arabic(d,(W//2,91),'الرجل الذي باع',font(46),IVORY)
    arabic(d,(W//2,147),'برج إيفل!',font(55),GOLD)
    d.line((79,179,W-79,179),fill=(194,158,100,200),width=1)
    arabic(d,(360,199),'إعداد وإخراج',font(22,'bold'),IVORY)
    centered(d,(209,199),'Shaaib Beshr',font(19,'inter'),GOLD)
    return rgba_to_cv(im)

def notice_card():
    im=Image.new('RGBA',(W,143),(0,0,0,0));d=ImageDraw.Draw(im)
    d.rounded_rectangle((28,9,W-28,132),5,fill=(16,20,21,228),outline=(201,166,105,190),width=1)
    d.rectangle((W-42,25,W-36,116),fill=RED)
    arabic(d,(W//2,41),'قصة متداولة.. وتفاصيل البيع',font(25,'bold'),IVORY)
    arabic(d,(W//2,74),'غير مؤكدة تاريخياً',font(27),GOLD)
    arabic(d,(W//2,107),'المشاهد المصوّرة توضيحية',font(19,'regular'),IVORY)
    return rgba_to_cv(im)

def header():
    im=Image.new('RGBA',(245,43),(0,0,0,0));d=ImageDraw.Draw(im)
    d.rounded_rectangle((0,0,244,42),3,fill=(17,21,22,205))
    d.rectangle((232,7,237,35),fill=RED)
    arabic(d,(118,21),'وثائقي للمعرفة',font(21,'bold'),IVORY)
    return rgba_to_cv(im)

def paper_sticker(main,sub='',w=330,h=100,rotate=0):
    im=Image.new('RGBA',(w+30,h+30),(0,0,0,0));d=ImageDraw.Draw(im)
    d.polygon([(12,8),(w+7,8),(w+18,17),(w+16,h+15),(17,h+17),(8,h+9)],fill=(8,10,11,120))
    d.polygon([(6,4),(w-1,4),(w+9,13),(w+8,h+8),(11,h+10),(3,h+3)],fill=(231,218,190,247))
    d.rectangle((w-26,5,w-12,h+7),fill=RED)
    arabic(d,(w//2-4, h//2-4 if not sub else h//2-17),main,font(31 if len(main)<18 else 26),INK)
    if sub:arabic(d,(w//2-4,h-15),sub,font(19,'bold'),(130,44,49))
    if rotate:im=im.rotate(rotate,resample=Image.Resampling.BICUBIC,expand=True)
    return rgba_to_cv(im)

def stamp_sticker(text,w=260,h=90,rotate=7):
    im=Image.new('RGBA',(w+26,h+26),(0,0,0,0));d=ImageDraw.Draw(im)
    d.rounded_rectangle((5,5,w+10,h+10),7,fill=(20,24,24,222),outline=(157,46,53,255),width=5)
    d.rounded_rectangle((12,12,w+3,h+3),4,outline=(157,46,53,230),width=2)
    arabic(d,((w+15)//2,(h+15)//2),text,font(35 if len(text)<12 else 28),GOLD)
    return rgba_to_cv(im.rotate(rotate,Image.Resampling.BICUBIC,expand=True))

def year_sticker(year,subtitle):
    im=Image.new('RGBA',(255,145),(0,0,0,0));d=ImageDraw.Draw(im)
    d.polygon([(8,8),(242,8),(249,18),(249,130),(19,137),(5,125)],fill=(23,27,28,232),outline=(196,156,99,255),width=2)
    centered(d,(128,58),year,font(56,'inter'),GOLD)
    arabic(d,(128,111),subtitle,font(23,'bold'),IVORY)
    return rgba_to_cv(im)

def portrait_sticker():
    im=Image.new('RGBA',(325,260),(0,0,0,0));d=ImageDraw.Draw(im)
    d.rectangle((13,13,318,249),fill=(0,0,0,115))
    d.polygon([(3,2),(310,2),(316,9),(315,242),(3,242)],fill=(230,217,190,255))
    photo=Image.open(AS/'archives/lustig_file.jpg').convert('RGB').crop((16,49,288,227))
    photo=photo.resize((282,185),Image.Resampling.LANCZOS)
    im.paste(photo,(16,14))
    d=ImageDraw.Draw(im)
    arabic(d,(155,221),'صورة أرشيفية · 1931',font(20,'bold'),(64,52,43))
    return rgba_to_cv(im.rotate(-2,Image.Resampling.BICUBIC,expand=True))

HEADER=header();TITLE=title_card();NOTICE=notice_card();PORTRAIT=portrait_sticker()
PAPER={
  'hotel':paper_sticker('فندق كريون','باريس',w=300,rotate=-2),
  'buyer':paper_sticker('أندريه بواسون','تصوير توضيحي',w=325,rotate=2),
  'money':paper_sticker('مبلغ إضافي','الصفقة المشبوهة',w=315,rotate=-2),
  'travel':paper_sticker('من باريس إلى فيينا','فرار بعد الصفقة',w=340,rotate=2),
  'repeat':paper_sticker('المحاولة الثانية','الشك يغيّر كل شيء',w=350,rotate=-2),
  'prison':paper_sticker('سجن ألكاتراز','الولايات المتحدة',w=325,rotate=2),
  'question':paper_sticker('كيف حدث ذلك؟','سؤال يستحق التفكير',w=330,rotate=-2),
  'last':paper_sticker('تحقّق قبل أن تثق','الفرصة ليست دائماً كما تبدو',w=365,rotate=2),
}
STAMPS={
  'sale':stamp_sticker('للبيع؟'),
  'fake':stamp_sticker('وثائق مزيفة',w=300),
  'secret':stamp_sticker('سري للغاية',w=275),
  'deal':stamp_sticker('صفقة وهمية',w=300),
}
YEARS={'1925':year_sticker('1925','باريس'), '1935':year_sticker('1935','القبض عليه'), '1947':year_sticker('1947','11 مارس')}

# Storyboard: each paragraph is keyed to the user's words, not arbitrary fixed time.
# Long paragraphs get a second image at a word boundary.
specs=[
 ('scene_eiffel','scene_impostor'),('scene_papers','scene_papers'),('scene_impostor','scene_impostor'),
 ('scene_eiffel','scene_papers'),('scene_blueprint','scene_blueprint'),('scene_eiffel','scene_eiffel'),
 ('scene_papers','scene_meeting'),('scene_hotel','scene_hotel'),('scene_meeting','scene_blueprint'),
 ('scene_papers','scene_papers'),('scene_meeting','scene_papers'),('scene_eiffel','scene_eiffel'),
 ('scene_buyer','scene_buyer'),('scene_buyer','scene_deal'),('scene_impostor','scene_impostor'),
 ('scene_deal','scene_buyer'),('scene_buyer','scene_impostor'),('scene_deal','scene_deal'),
 ('scene_deal','scene_escape'),('scene_papers','scene_papers'),('scene_buyer','scene_buyer'),
 ('scene_impostor','scene_impostor'),('scene_eiffel','scene_eiffel'),('scene_buyer','scene_papers'),
 ('scene_escape','scene_escape'),('scene_papers','scene_papers'),('scene_papers','scene_impostor'),
 ('scene_alcatraz','scene_alcatraz'),('scene_alcatraz','scene_papers'),('scene_eiffel','scene_eiffel'),
 ('scene_papers','scene_papers'),('scene_eiffel','scene_eiffel'),('scene_impostor','scene_buyer'),
 ('scene_papers','scene_papers'),('scene_deal','scene_eiffel'),
]
assert len(specs)==len(PARAS)==35
scenes=[]
for p in PARAS:
    idx=p['index']; start=p['start']; end=PARAS[idx+1]['start'] if idx+1<len(PARAS) else DURATION
    primary,second=specs[idx]
    scenes.append({'start':start,'asset':primary,'paragraph':idx})
    if end-start>12.0:
        mid=start+(end-start)*0.51
        candidates=[x['start'] for x in WORDS if x['paragraph']==idx and x['start']>start+4 and x['start']<end-4]
        if candidates:
            mid=min(candidates,key=lambda t:abs(t-mid))
            scenes.append({'start':mid,'asset':second,'paragraph':idx})
scenes.sort(key=lambda x:x['start'])
for i,s in enumerate(scenes):s['end']=scenes[i+1]['start'] if i+1<len(scenes) else DURATION
SCENE_STARTS=[s['start'] for s in scenes]
(P/'storyboard.json').write_text(json.dumps(scenes,ensure_ascii=False,indent=2),encoding='utf8')

# Sparse meaningful insert cards; they enter and leave gently, not on every sentence.
# Values chosen at actual words of the provided recording.
events=[
 (26.2,30.5,'year','1925',30,128),
 (30.6,38.7,'portrait','real',28,146),
 (53.25,56.4,'stamp','sale',268,160),
 (64.8,70.8,'stamp','fake',252,142),
 (77.5,84.2,'paper','hotel',32,135),
 (104.95,112.8,'stamp','secret',252,145),
 (131.8,138.8,'paper','buyer',33,145),
 (155.6,164.8,'paper','money',31,144),
 (179.1,184.8,'stamp','deal',248,137),
 (195.5,201.9,'paper','travel',29,150),
 (218.0,227.0,'paper','repeat',30,145),
 (248.0,253.8,'year','1935',30,135),
 (259.2,266.4,'paper','prison',30,137),
 (268.8,275.6,'year','1947',30,135),
 (293.0,300.9,'paper','question',35,142),
 (320.0,332.5,'paper','last',30,144),
]

# Compose subtitles in compact, fully-shaped RTL word groups.
# Gold highlights track the active word; no Arabic letters are individually animated.
measure=ImageDraw.Draw(Image.new('RGB',(1,1)))
cap_font=font(36)
def textwidth(s):
    return float(measure.textlength(s,font=cap_font,direction='rtl',language='ar'))

groups=[];current=[];curwidth=0
for idx,word in enumerate(WORDS):
    width=textwidth(word['word']); gap=13
    split=False
    if current:
        prev=WORDS[current[-1]]
        split=(len(current)>=4 or curwidth+gap+width>454 or prev['paragraph']!=word['paragraph']
               or word['start']-prev['end']>0.65 or prev['word'].endswith(('!','؟',':','...')))
    if split:
        groups.append(current);current=[];curwidth=0
    curwidth+=width+(gap if current else 0);current.append(idx)
if current:groups.append(current)
GROUP_STARTS=[WORDS[g[0]]['start'] for g in groups]

@lru_cache(maxsize=28)
def caption(group_index,active_index):
    inds=groups[group_index]; entries=[WORDS[j]['word'] for j in inds]
    widths=[textwidth(s) for s in entries];gap=13
    total=sum(widths)+gap*(len(widths)-1)
    im=Image.new('RGB',(516,75),(18,22,23));d=ImageDraw.Draw(im)
    d.rectangle((0,0,515,74),outline=(91,80,64),width=1)
    d.rectangle((509,8,514,66),fill=RED)
    d.line((10,72,490,72),fill=(111,92,67),width=1)
    x=(516+total)/2-2
    for k,(entry,width) in enumerate(zip(entries,widths)):
        x-=width
        fill=GOLD if k==active_index else IVORY
        # anchor la makes the horizontal metrics agree with textlength and RAQM.
        d.text((round(x),7),entry,font=cap_font,fill=fill,direction='rtl',language='ar',stroke_width=0)
        if k==active_index:d.line((round(x),62,round(x+width),62),fill=GOLD,width=2)
        x-=gap
    return cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)

@lru_cache(maxsize=12)
def image_for(name):
    path=AS/(name+'.png')
    im=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if im is None:raise RuntimeError('Missing scene image '+str(path))
    return im

def backdrop(s,t,index):
    src=image_for(s['asset']); sh,sw=src.shape[:2]
    span=max(.01,s['end']-s['start']);u=max(0,min(1,(t-s['start'])/span));
    # Varied camera direction, slow physical drift and restrained zoom.
    sign=1 if index%2==0 else -1
    ease=smooth(u)
    zoom=(1.045+0.07*(ease if index%3!=0 else 1-ease))
    scale=max(W/sw,H/sh)*zoom
    ex=sw*scale-W;ey=sh*scale-H
    dx=-ex*(.5 + sign*.20*(2*ease-1));dy=-ey*(.49 + (1 if index%4<2 else -1)*.15*(2*ease-1))
    mat=np.array([[scale,0,dx],[0,scale,dy]],dtype=np.float32)
    return cv2.warpAffine(src,mat,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)

def alpha_overlay(dst,im,x,y,opacity=1.0):
    if opacity<=0:return
    ih,iw=im.shape[:2];x=int(x);y=int(y)
    left=max(0,x);top=max(0,y);right=min(W,x+iw);bottom=min(H,y+ih)
    if right<=left or bottom<=top:return
    src=im[top-y:bottom-y,left-x:right-x];dstroi=dst[top:bottom,left:right]
    a=src[:,:,3:4].astype(np.float32)*(opacity/255.0)
    # NumPy broadcasting is limited to the small sticker region.
    dstroi[:]=np.clip(src[:,:,:3].astype(np.float32)*a+dstroi.astype(np.float32)*(1-a),0,255).astype(np.uint8)

def make_frame(t):
    si=max(0,bisect.bisect_right(SCENE_STARTS,t)-1);scene=scenes[si]
    frame=backdrop(scene,t,si)
    # A paper cut, soft crossfade or horizontal reveal; rhythm varies scene to scene.
    if si>0 and t<scene['start']+.44:
        p=smooth((t-scene['start'])/.44)
        old=backdrop(scenes[si-1],max(scenes[si-1]['start'],scene['start']-.03),si-1)
        if si%3==0:
            frame=cv2.addWeighted(frame,p,old,1-p,0)
        elif si%3==1:
            cut=int(W*p); merged=old
            if cut>0:merged[:,W-cut:]=frame[:,W-cut:]
            if 0<W-cut<W:cv2.line(merged,(W-cut,0),(W-cut,H),(100,140,190),3)
            frame=merged
        else:
            cut=int(H*p);merged=old
            if cut>0:merged[:cut,:]=frame[:cut,:]
            if 0<cut<H:cv2.line(merged,(0,cut),(W,cut),(100,140,190),3)
            frame=merged
    # Persistent small identity appears after the opening cover.
    if t>=6.8:alpha_overlay(frame,HEADER,W-260,24,min(1,(t-6.8)/.35))
    if t<7.2:
        a=smooth(t/.28)*(1-smooth((t-6.5)/.7))
        alpha_overlay(frame,TITLE,0,46,a)
    if 19.0<=t<=25.65:
        alpha_overlay(frame,NOTICE,0,115,smooth((t-19.0)/.35)*(1-smooth((t-25.2)/.45)))
    for start,end,kind,key,x,y in events:
        if start<=t<end:
            a=smooth((t-start)/.35)*(1-smooth((t-(end-.4))/.4))
            offset=(1-smooth((t-start)/.4))*22
            if kind=='year':im=YEARS[key]
            elif kind=='stamp':im=STAMPS[key]
            elif kind=='paper':im=PAPER[key]
            else:im=PORTRAIT
            alpha_overlay(frame,im,x,y+offset,a)
    # Caption: all words of the group are readable, with the spoken word highlighted.
    gi=bisect.bisect_right(GROUP_STARTS,t)-1
    if gi>=0:
        inds=groups[gi]; first=WORDS[inds[0]];last=WORDS[inds[-1]]
        next_start=GROUP_STARTS[gi+1] if gi+1<len(groups) else DURATION
        until=min(next_start,last['end']+.40)
        if t>=first['start'] and t<until:
            active=0
            for j,ind in enumerate(inds):
                if t>=WORDS[ind]['start']-.015:active=j
                else:break
            block=caption(gi,active)
            frame[790:865,30:546]=block
    return frame

def make_sfx(path):
    from scipy.signal import butter, sosfilt
    rate=24000; rng=np.random.default_rng(52025)
    data=np.zeros(round(DURATION*rate),dtype=np.float32)
    positions=[26.2,53.25,64.8,77.5,104.95,131.8,155.6,179.1,195.5,218,248,259.2,268.8,293,320]
    for n,at in enumerate(positions):
        dur=.24 if n%2 else .31; L=int(dur*rate)
        noise=rng.standard_normal(L).astype(np.float32)
        noise=sosfilt(butter(3,[320,3300],btype='bandpass',fs=rate,output='sos'),noise).astype(np.float32)
        env=np.sin(np.linspace(0,math.pi,L))**2
        effect=noise*env*.020
        if n in (1,7,10,12):
            tt=np.arange(L)/rate
            effect+=.045*np.sin(2*math.pi*76*tt)*np.exp(-tt*16)
        pos=max(0,int(at*rate));avail=min(L,len(data)-pos)
        if avail>0:data[pos:pos+avail]+=effect[:avail]
    data=np.clip(data,-.30,.30)
    with wave.open(str(path),'wb') as wf:
        wf.setparams((1,2,rate,0,'NONE','not compressed'))
        wf.writeframes((data*32767).astype('<i2').tobytes())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--start',type=float,default=0.0)
    parser.add_argument('--duration',type=float,default=None);parser.add_argument('--output',default='output/Eiffel_Documentary_Shaaib_Beshr.mp4')
    parser.add_argument('--preset',default='medium');parser.add_argument('--bitrate',default='390k');args=parser.parse_args()
    duration=min(DURATION-args.start,args.duration if args.duration is not None else DURATION-args.start)
    out=P/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    fx=P/'sfx.wav'
    if not fx.exists():make_sfx(fx)
    command=['ffmpeg','-hide_banner','-loglevel','warning','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0',
             '-ss',str(args.start),'-i',str(P/'narration.wav'),'-ss',str(args.start),'-i',str(fx),
             '-filter_complex','[1:a]highpass=f=60,loudnorm=I=-16:TP=-1.5:LRA=7,aresample=24000[voice];[2:a]volume=0.8,aresample=24000[effects];[voice][effects]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.84:level=false[a]',
             '-map','0:v','-map','[a]','-t',f'{duration:.3f}','-c:v','libx264','-preset',args.preset,'-b:v',args.bitrate,'-maxrate','800k','-bufsize','1600k',
             '-g','96','-pix_fmt','yuv420p','-profile:v','high','-level','3.1','-c:a','aac','-b:a','48k','-ar','24000','-ac','1','-movflags','+faststart',str(out)]
    print('Rendering',round(duration,2),'seconds to',out,'scenes',len(scenes),'caption groups',len(groups),flush=True)
    log=open(P/'render_ffmpeg.log','w',encoding='utf8')
    proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,bufsize=10**7)
    frames=round(duration*FPS);start_time=time.time()
    try:
        for n in range(frames):
            t=args.start+n/FPS
            im=make_frame(t)
            proc.stdin.write(im.tobytes())
            if n%(FPS*12)==0:
                elapsed=time.time()-start_time
                print(f'{n/FPS:6.1f}/{duration:.1f}s  render speed {n/max(.01,elapsed):.1f} fps',flush=True)
        proc.stdin.close();code=proc.wait()
    except BrokenPipeError:
        code=proc.wait()
    finally:log.close()
    if code:
        print((P/'render_ffmpeg.log').read_text(encoding='utf8')[-4000:],file=sys.stderr)
        raise SystemExit(code)
    print('DONE',out,'size',round(out.stat().st_size/1024/1024,2),'MiB, time',round(time.time()-start_time,1),'sec',flush=True)

if __name__=='__main__':main()
