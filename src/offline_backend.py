"""NumPy-only Haar DWT + selected DCT coefficient QIM research prototype.
This is NOT ShieldMnt's invisible-watermark implementation or WAM.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

DELTA=80.0
v=np.arange(8)+.5
basis=(np.sqrt(2/8)*np.cos(np.pi*v/8))[:,None]*(np.sqrt(2/8)*np.cos(2*np.pi*v/8))[None,:]

def ll_blocks(rgb):
    y=np.array(rgb,dtype=float) @ np.array([.299,.587,.114])
    ll=(y[::2,::2]+y[1::2,::2]+y[::2,1::2]+y[1::2,1::2])/2
    blocks=ll.reshape(16,8,16,8).transpose(0,2,1,3)
    return blocks

class WatermarkEncoder:
    def set_watermark(self, kind, packet):self.bits=np.asarray(packet)
    def encode(self,bgr,method):
        rgb=bgr[:,:,::-1].astype(float)
        blocks=ll_blocks(rgb)
        coeff=(blocks*basis).sum(axis=(2,3))
        bits=np.resize(self.bits,256).reshape(16,16)
        target=(np.floor(coeff/DELTA)+.25+.5*bits)*DELTA
        change=(target-coeff)[:,:,None,None]*basis
        change=change.transpose(0,2,1,3).reshape(128,128)
        spatial=np.repeat(np.repeat(change/2,2,axis=0),2,axis=1)
        return np.clip(np.rint(rgb+spatial[:,:,None]),0,255).astype(np.uint8)[:,:,::-1]

class WatermarkDecoder:
    def __init__(self,kind,length):self.length=length
    def decode(self,bgr,method):
        coeff=(ll_blocks(bgr[:,:,::-1])*basis).sum(axis=(2,3)).ravel()
        # Circular distance difference provides bounded soft evidence per bit.
        phase=(coeff/DELTA)%1
        d0=np.minimum(np.abs(phase-.25),1-np.abs(phase-.25))
        d1=np.minimum(np.abs(phase-.75),1-np.abs(phase-.75))
        return ((d0-d1).reshape(-1,self.length).sum(axis=0)>0).astype(np.uint8)

def peak_signal_noise_ratio(a,b,data_range=255):
    mse=np.mean((a.astype(float)-b.astype(float))**2)
    return 10*np.log10(data_range**2/mse) if mse else float('inf')

def box7(a):
    # Only valid windows; equivalent to cropping the SSIM border by 3 pixels.
    integral=np.pad(a,((1,0),(1,0),(0,0))).cumsum(axis=0).cumsum(axis=1)
    return (integral[7:,7:]-integral[:-7,7:]-integral[7:,:-7]+integral[:-7,:-7])/49

def structural_similarity(a,b,data_range=255,channel_axis=2):
    a=a.astype(float);b=b.astype(float)
    ux,uy=box7(a),box7(b)
    vx=(box7(a*a)-ux*ux)*49/48
    vy=(box7(b*b)-uy*uy)*49/48
    vxy=(box7(a*b)-ux*uy)*49/48
    c1=(.01*data_range)**2;c2=(.03*data_range)**2
    s=((2*ux*uy+c1)*(2*vxy+c2))/((ux*ux+uy*uy+c1)*(vx+vy+c2))
    return s.mean()

NAMES=['smooth_gradient','color_waves','fine_texture','coarse_texture','checkerboard',
       'geometric_shapes','text_document','landscape_drawing','radial_pattern','mixed_texture']

def fixture(name):
    i=NAMES.index(name);rng=np.random.default_rng(100+i)
    h,w=512,1024;y,x=np.mgrid[:h,:w];xx=x/w;yy=y/h
    if i==0:a=np.stack([40+170*xx,50+150*yy,60+100*(xx+yy)/2],2)
    elif i==1:a=np.stack([128+70*np.sin(xx*18),128+70*np.sin(yy*23),128+70*np.cos((xx+yy)*12)],2)
    elif i==2:a=100+55*rng.random((h,w,3))
    elif i==3:
        im=Image.fromarray(rng.integers(30,225,(32,64,3),dtype=np.uint8)).resize((w,h),Image.Resampling.BICUBIC)
        return np.array(im)
    elif i==4:
        z=((x//25+y//25)%2)*130+60;a=np.stack([z,.8*z+20,.65*z+30],2)
    elif i==5:
        im=Image.new('RGB',(w,h),(180,195,210));d=ImageDraw.Draw(im)
        for _ in range(90):
            xx0=int(rng.integers(0,w-100));yy0=int(rng.integers(0,h-80));color=tuple(rng.integers(25,230,3).tolist())
            d.ellipse((xx0,yy0,xx0+int(rng.integers(15,100)),yy0+int(rng.integers(15,80))),fill=color)
        return np.array(im.filter(ImageFilter.GaussianBlur(.6)))
    elif i==6:
        im=Image.new('RGB',(w,h),(225,225,215));d=ImageDraw.Draw(im)
        for yy0 in range(20,500,20):d.text((25,yy0),'Robust invisible watermarking: 64-bit payload, coding, image processing. '*2,fill=(45,45,50))
        return np.array(im)
    elif i==7:
        a=np.stack([100+50*yy,160+30*yy,220-40*yy],2)
        ground=y>280+50*np.sin(xx*11)
        a[ground]=np.stack([80+25*np.sin(xx*7),120+30*np.cos(xx*8),55+10*np.sin(yy*12)],2)[ground]
    elif i==8:
        z=128+70*np.sin(np.sqrt((xx-.5)**2+(yy-.5)**2)*80);a=np.stack([z,255-z,.8*z+30],2)
    else:
        z=60+120*xx+20*np.sin(yy*70)+rng.normal(0,8,(h,w));a=np.stack([z,z*.8+20,z*.6+45],2)
    return np.clip(np.rint(a),0,255).astype(np.uint8)

class Fixtures:
    def __getattr__(self,name):return lambda:fixture(name)
data=Fixtures()
