"""Bounded official-WAM setup diagnostics; never substitutes another model."""
import concurrent.futures, hashlib, importlib.metadata, json, platform, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LOG=ROOT/'logs';LOG.mkdir(exist_ok=True)
def run(name,command,timeout=40):
    start=time.time()
    try:
        p=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
        code=p.returncode;output=p.stdout+p.stderr
    except subprocess.TimeoutExpired as e:
        code=124;output='Subprocess timeout after '+str(timeout)+' seconds.\n'
        for part in [e.stdout,e.stderr]:
            if part:output+=part.decode(errors='replace') if isinstance(part,bytes) else part
    r={'name':name,'command':command,'exit_code':code,'seconds':round(time.time()-start,3),'output':output}
    (LOG/(name+'.json')).write_text(json.dumps(r,indent=2))
    print(name,code,output[-300:],flush=True)
    return r
def main():
    urls={
        'pypi':'https://pypi.org/simple/torch/',
        'github':'https://github.com/facebookresearch/watermark-anything.git',
        'official_checkpoint':'https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth',
        'checkpoint_mirror':'https://huggingface.co/facebook/watermark-anything/resolve/main/checkpoint.pth',
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(run,name,['curl','-I','-L','--connect-timeout','7','--max-time','12',url],20) for name,url in urls.items()]
        network=[j.result() for j in jobs]
    install=run('pytorch_install',[str(ROOT/'.venv/bin/python'),'-m','pip','install','--no-cache-dir','--disable-pip-version-check','--timeout','8','--retries','0','torch==2.5.1','torchvision==0.20.1'],35)
    packages=run('installed_packages',[str(ROOT/'.venv/bin/python'),'-m','pip','freeze','--all'],15)
    model_import=run('torch_import',[str(ROOT/'.venv/bin/python'),'-c','import torch, torchvision; print(torch.__version__, torchvision.__version__); print("CUDA",torch.cuda.is_available()); print("MPS",torch.backends.mps.is_available())'],15)
    result={'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
        'wam_commit':'2c08af04d037d5667c02f6ddebbda9ff04581c3e',
        'checkpoint_url':urls['official_checkpoint'],'checkpoint_downloaded':False,'checkpoint_sha256':None,
        'network':network,'installation':install,'packages':packages,'torch_import':model_import,
        'phase0_ready':False,'inference_executed':False}
    (ROOT/'environment_probe.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
