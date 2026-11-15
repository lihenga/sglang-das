"""Durable container-side service process groups, scoped to this experiment."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/public/home/starrkk/xunfei/bench/1113/l3_128_apply_ec45640d_20261113')
def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('run');a.add_argument('role');a.add_argument('script');x=a.parse_args()
    folder=ROOT/'runs'/x.run;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/(x.role+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=folder/(x.role+'.json');assert not receipt.exists(),str(receipt)
    env=os.environ.copy()
    for k in list(env):
        # Explicit launch configuration; config*.sh validates 64/128/256.
        if k == 'SGLANG_HICACHE_DIRECT_STORAGE_PAGE_SIZE':
            continue
        if k.startswith(('SGLANG_','SPARK_','MOONCAKE_','MC_','DEEPEP_','NCCL_','GLOO_','ROCSHMEM_')) or k in ['PYTHONPATH','PYTHONHOME','LD_PRELOAD','CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']:
            env.pop(k,None)
    env.update(RUN_ID=x.run,DEPLOY_COMPONENT=x.role,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    with (folder/(x.role+'.launcher.log')).open('a') as out:
        p=subprocess.Popen(['bash',str(ROOT/x.script)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    ticks=Path('/proc',str(p.pid),'stat').read_text().split(') ',1)[1].split()[19]
    save(receipt,dict(role=x.role,pid=p.pid,start_ticks=ticks,run=x.run,script=x.script,container='xunfei-1110rc1',started_at=datetime.datetime.now().isoformat()))
    rc=p.wait();save(folder/(x.role+'.exit.json'),dict(returncode=rc,finished_at=datetime.datetime.now().isoformat()))
