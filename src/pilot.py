"""Small, paired, registration-assisted watermark coding experiment.

Run from project root: .venv/bin/python src/pilot.py
Decoder never receives the true source payload or expected packet bits.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.abspath('.cache/matplotlib'))
os.environ.setdefault('SKIMAGE_DATADIR', os.path.abspath('data/skimage'))
import csv, hashlib, io, itertools, json, platform, time
from pathlib import Path
import numpy as np
from PIL import Image
OFFLINE = os.environ.get('PILOT_OFFLINE') == '1'
if OFFLINE:
    from offline_backend import data, peak_signal_noise_ratio, structural_similarity, WatermarkEncoder, WatermarkDecoder
else:
    from skimage import data
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from imwatermark import WatermarkEncoder, WatermarkDecoder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
T = 256
SEED = 20261004
METHODS = ['raw_padded', 'repetition', 'rlnc', 'mds']
NAMES = ['astronaut', 'coffee', 'chelsea', 'rocket', 'camera',
         'hubble_deep_field', 'immunohistochemistry', 'brick', 'grass', 'gravel']
if OFFLINE:
    from offline_backend import NAMES

# GF(256), polynomial x^8+x^4+x^3+x^2+1.
EXP = np.zeros(512, dtype=np.uint16)
LOG = np.zeros(256, dtype=np.int16)
x = 1
for i in range(255):
    EXP[i], LOG[x] = x, i
    x <<= 1
    if x & 256: x ^= 0x11d
EXP[255:] = EXP[np.arange(257) % 255]
MUL = np.zeros((256, 256), dtype=np.uint8)
for a in range(1, 256):
    for b in range(1, 256): MUL[a, b] = EXP[int(LOG[a]) + int(LOG[b])]

def matmul(a, b):
    return np.bitwise_xor.reduce(MUL[a[:, :, None], b[None, :, :]], axis=1)

def solve(a, b):
    """Return source matrix only for rank 4 and consistent equations."""
    if len(a) < 4: return None
    aug = np.concatenate([np.asarray(a, dtype=np.uint8), np.asarray(b, dtype=np.uint8)], axis=1)
    row = 0
    for col in range(4):
        piv = next((j for j in range(row, len(aug)) if aug[j, col]), None)
        if piv is None: return None
        aug[[row, piv]] = aug[[piv, row]]
        inv = int(EXP[255 - int(LOG[aug[row, col]])])
        aug[row] = MUL[inv, aug[row]]
        for j in range(len(aug)):
            if j != row and aug[j, col]: aug[j] ^= MUL[aug[j, col], aug[row]]
        row += 1
    if np.any(aug[4:, 4:]): return None
    return aug[:4, 4:].copy()

def crc8(buf):
    c = 0
    for val in buf:
        c ^= int(val)
        for _ in range(8): c = ((c << 1) ^ (0x07 if c & 128 else 0)) & 255
    return c

IDENTITY = np.eye(4, dtype=np.uint8)
MATRICES = {
    'raw_padded': np.concatenate([IDENTITY, np.zeros((4, 4), dtype=np.uint8)]),
    'repetition': IDENTITY[[0, 1, 2, 3, 2, 3, 0, 1]],
    'rlnc': np.concatenate([IDENTITY, np.random.default_rng(SEED).integers(0, 256, (4, 4), dtype=np.uint8)]),
    'mds': np.array([[1, x, MUL[x,x], MUL[MUL[x,x],x]] for x in range(1, 9)], dtype=np.uint8),
}

def encode_packets(source, method):
    values = matmul(MATRICES[method], source)
    packets = []
    for i, value in enumerate(values):
        header = bytes([i, int(value[0]), int(value[1])])
        packets.append(np.unpackbits(np.frombuffer(header + bytes([crc8(header)]), dtype=np.uint8)))
    return packets

def parse(bits):
    b = np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()
    if len(b) != 4 or b[0] >= 8 or crc8(b[:3]) != b[3]: return None
    return b[0], np.frombuffer(b[1:3], dtype=np.uint8)

def recover(bits_list, method):
    accepted = {}
    for bits in bits_list:
        packet = parse(bits)
        if packet is None: continue
        idx, val = packet
        if idx in accepted and not np.array_equal(accepted[idx], val): return None
        accepted[idx] = val
    indices = sorted(accepted)
    return solve(MATRICES[method][indices], [accepted[i] for i in indices])

def tile_box(i):
    r, c = divmod(i, 4)
    return c*T, r*T, (c+1)*T, (r+1)*T

def embed(im, packets):
    arr = np.array(im)
    for i, packet in enumerate(packets):
        x0,y0,x1,y1 = tile_box(i)
        enc = WatermarkEncoder()
        enc.set_watermark('bits', packet)
        arr[y0:y1,x0:x1] = enc.encode(arr[y0:y1,x0:x1,::-1].copy(), 'dwtDct')[:,:,::-1]
    return arr

def decode_tiles(arr):
    dec = WatermarkDecoder('bits', 32)
    out = {}
    for i in range(8):
        x0,y0,x1,y1 = tile_box(i)
        out[i] = dec.decode(arr[y0:y1,x0:x1,::-1].copy(), 'dwtDct').astype(np.uint8)
    return out

def jpeg(arr, q):
    buf=io.BytesIO()
    Image.fromarray(arr).save(buf,format='JPEG',quality=q,subsampling=2)
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'))

def score(bits, packets, source, method):
    recovered = recover(list(bits.values()), method)
    valid = [parse(b) for b in bits.values()]
    return {
        'success': int(recovered is not None and np.array_equal(recovered, source)),
        'wrong_payload': int(recovered is not None and not np.array_equal(recovered, source)),
        'retained_tiles': len(bits),
        'accepted_packets': sum(p is not None for p in valid),
        'wrong_accepted_packets': sum(p is not None and not np.array_equal(bits[i],packets[i]) for i,p in zip(bits,valid)),
        'packet_ber': float(np.mean([np.mean(b != packets[i]) for i,b in bits.items()])) if bits else None,
    }

def load_images():
    out=[]
    manifest=[]
    for name in NAMES:
        arr=getattr(data,name)()
        im=Image.fromarray(arr).convert('RGB').resize((1024,512),Image.Resampling.LANCZOS)
        path=ROOT/'data'/f'{name}.png'
        im.save(path)
        out.append((name,np.array(im)))
        manifest.append({'name':name,'original_shape':list(arr.shape),'resized_shape':[512,1024,3],
                         'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source':f'offline_backend.fixture.{name}' if OFFLINE else f'skimage.data.{name}'})
    (OUT/'image_manifest.json').write_text(json.dumps(manifest,indent=2))
    return out

def coding_checks():
    rng=np.random.default_rng(101)
    for method in METHODS:
        for _ in range(20):
            source=rng.integers(0,256,(4,2),dtype=np.uint8)
            packets=encode_packets(source,method)
            assert np.array_equal(recover(packets,method),source)
            for p in packets:
                for bit in range(32):
                    q=p.copy();q[bit]^=1
                    assert parse(q) is None
    coding=[]
    source=rng.integers(0,256,(4,2),dtype=np.uint8)
    for method in METHODS:
        packets=encode_packets(source,method)
        for m in range(1,9):
            subsets=list(itertools.combinations(range(8),m))
            good=sum(np.array_equal(recover([packets[i] for i in inds],method),source) for inds in subsets)
            coding.append({'method':method,'retained_packets':m,'successes':good,'total':len(subsets),'rate':good/len(subsets)})
    write_csv(OUT/'coding_only.csv',coding)
    return coding

def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def summarize(rows, qualities):
    attacks=list(dict.fromkeys(r['attack'] for r in rows))
    summary=[]; contrasts=[]
    rng=np.random.default_rng(879)
    for attack in attacks:
        per_image={}
        for method in METHODS:
            rr=[r for r in rows if r['attack']==attack and r['method']==method]
            image_means=np.array([np.mean([r['success'] for r in rr if r['image']==n]) for n in NAMES])
            per_image[method]=image_means
            boot=image_means[rng.integers(0,len(NAMES),(2000,len(NAMES)))].mean(axis=1)
            summary.append({'attack':attack,'method':method,'n':len(rr),'successes':sum(r['success'] for r in rr),
                'recovery_rate':float(image_means.mean()),'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),
                'packet_ber':float(np.mean([r['packet_ber'] for r in rr if r['packet_ber'] is not None])),
                'packet_acceptance':sum(r['accepted_packets'] for r in rr)/sum(r['retained_tiles'] for r in rr),
                'wrong_accepted_packets':sum(r['wrong_accepted_packets'] for r in rr),'wrong_payloads':sum(r['wrong_payload'] for r in rr)})
        for method in ['rlnc','mds']:
            delta=per_image[method]-per_image['repetition']
            boot=delta[rng.integers(0,len(NAMES),(2000,len(NAMES)))].mean(axis=1)
            contrasts.append({'attack':attack,'method':method,'baseline':'repetition','delta':float(delta.mean()),
                'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975))})
    qsummary=[]
    for method in METHODS:
        qs=[q for q in qualities if q['method']==method]
        qsummary.append({'method':method,'psnr':float(np.mean([q['psnr'] for q in qs])),
                         'ssim':float(np.mean([q['ssim'] for q in qs])),
                         'embed_seconds':float(np.mean([q['embed_seconds'] for q in qs]))})
    write_csv(OUT/'attack_summary.csv',summary)
    write_csv(OUT/'paired_differences.csv',contrasts)
    write_csv(OUT/'quality_summary.csv',qsummary)
    result={'config':{'backend':'NumPy Haar-DWT + DCT-QIM prototype' if OFFLINE else 'invisible-watermark dwtDct',
        'dataset':'10 procedural synthetic fixtures' if OFFLINE else '10 skimage example images',
        'seed':SEED,'K':4,'N':8,'payload_bits':64,'packet_bits':32,'image_count':len(NAMES),'payloads_per_image':3,
        'tile_size':256,'grid':[2,4],'crop_registration':'known coordinates; partial tiles discarded',
        'matrices':{k:v.tolist() for k,v in MATRICES.items()}},'quality':qsummary,'attacks':summary,'contrasts':contrasts}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    return result

def main():
    start=time.time()
    coding_checks()
    images=load_images()
    rows=[];qualities=[];negative=[]
    four=[[0,1,2,3],[4,5,6,7],[0,1,4,5],[1,2,5,6],[2,3,6,7]]
    six=[[0,1,2,4,5,6],[1,2,3,5,6,7]]
    for ni,(name,original) in enumerate(images):
        unmarked=decode_tiles(original)
        negative.append({'image':name,'accepted_packets':sum(parse(b) is not None for b in unmarked.values()),
                         **{f'{m}_payload_output':int(recover(list(unmarked.values()),m) is not None) for m in METHODS}})
        for trial in range(3):
            seed=SEED+ni*100+trial
            rng=np.random.default_rng(seed)
            source=rng.integers(0,256,(4,2),dtype=np.uint8)
            # Fixed positions independent of method. Actual subimage is represented by
            # retaining complete tile pixels; crop offsets are supplied by the evaluator.
            random_crops=[]
            h,w=444,886 # 75.03% retained area; random offset, no resizing.
            for _ in range(8):
                x0=int(rng.integers(0,1024-w+1));y0=int(rng.integers(0,512-h+1))
                inds=[i for i in range(8) if (lambda b:b[0]>=x0 and b[1]>=y0 and b[2]<=x0+w and b[3]<=y0+h)(tile_box(i))]
                random_crops.append(inds)
            for method in METHODS:
                packets=encode_packets(source,method)
                t0=time.time();wm=embed(original,packets);elapsed=time.time()-t0
                qualities.append({'image':name,'trial':trial,'method':method,'psnr':float(peak_signal_noise_ratio(original,wm,data_range=255)),
                    'ssim':float(structural_similarity(original,wm,data_range=255,channel_axis=2)), 'embed_seconds':elapsed})
                if ni==0 and trial==0:
                    Image.fromarray(original).save(OUT/'example_original.png')
                    Image.fromarray(wm).save(OUT/f'example_{method}.png')
                clean_bits=None
                for attack in ['clean','jpeg75','jpeg50','noise2','noise5','resize075','rotate2']:
                    if attack=='clean':attacked=wm
                    elif attack.startswith('jpeg'):attacked=jpeg(wm,int(attack[4:]))
                    elif attack.startswith('noise'):
                        noise=np.random.default_rng(seed+993).normal(0,int(attack[5:]),wm.shape)
                        attacked=np.clip(np.rint(wm.astype(float)+noise),0,255).astype(np.uint8)
                    elif attack=='resize075':
                        attacked=np.array(Image.fromarray(wm).resize((768,384),Image.Resampling.BICUBIC).resize((1024,512),Image.Resampling.BICUBIC))
                    else:attacked=np.array(Image.fromarray(wm).rotate(2,resample=Image.Resampling.BICUBIC,expand=False,fillcolor=(0,0,0)))
                    bits=decode_tiles(attacked)
                    base={'image':name,'trial':trial,'method':method}
                    rows.append({**base,'attack':attack,'variant':0,**score(bits,packets,source,method)})
                    if attack=='clean':clean_bits=bits
                    if attack in ['clean','jpeg75','jpeg50']:
                        for count,windows in [(4,four),(6,six)]:
                            for j,inds in enumerate(windows):
                                rows.append({**base,'attack':f'crop{count}/8'+('' if attack=='clean' else '+'+attack),'variant':j,
                                    **score({i:bits[i] for i in inds},packets,source,method)})
                    if attack=='clean':
                        for j,inds in enumerate(random_crops):
                            rows.append({**base,'attack':'offset_crop75_oracle','variant':j,**score({i:bits[i] for i in inds},packets,source,method)})
                print(f'{name} payload={trial} method={method} done',flush=True)
        write_csv(OUT/'trials.csv',rows)
        write_csv(OUT/'quality.csv',qualities)
        write_csv(OUT/'negative_controls.csv',negative)
    result=summarize(rows,qualities)
    (OUT/'runtime.json').write_text(json.dumps({'seconds':time.time()-start,'python':platform.python_version(),'platform':platform.platform()},indent=2))
    print(json.dumps({'quality':result['quality'],'contrasts':result['contrasts']},indent=2),flush=True)

if __name__=='__main__':main()
