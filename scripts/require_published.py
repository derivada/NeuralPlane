"""Require remotely published source, private baseline and committed harness pin."""
import json
import os
from pathlib import Path
import subprocess

def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()

def main():
    repo=Path(__file__).resolve().parents[1]
    harness=Path(os.environ.get('PRIMIMIX_EVAL_ROOT',str(Path.home()/'primimix-eval'))).resolve()
    head=git(repo,'rev-parse','HEAD')
    name='neuralplane' if (repo/'neuralplane').is_dir() else 'plana3r'
    remote=git(repo,'ls-remote','origin','refs/heads/dilab-eval-fairness').split()
    if not remote or remote[0]!=head:
        raise RuntimeError('baseline branch must be pushed before setup/testing')
    if git(repo,'status','--porcelain','--untracked-files=no'):
        raise RuntimeError('baseline has tracked source changes')
    pin=git(harness,'ls-tree','HEAD',f'submodules/{name}').split()
    if len(pin)<3 or pin[2]!=head:
        raise RuntimeError('harness must commit the exact baseline pin before setup/testing')
    branch=git(harness,'branch','--show-current')
    published=git(harness,'ls-remote','origin',f'refs/heads/{branch}').split()
    if not published or published[0]!=git(harness,'rev-parse','HEAD'):
        raise RuntimeError('harness integration must be pushed before setup/testing')
    subprocess.run(['git','-C',str(harness),'diff','--exit-code','HEAD','--',
                    'scripts','primimix_eval','conf','.gitmodules',f'submodules/{name}'],check=True)
    repository='derivada/NeuralPlane' if name=='neuralplane' else 'derivada/plana3r'
    metadata=json.loads(subprocess.check_output(['gh','api',f'repos/{repository}'],text=True))
    if metadata.get('private') is not True:
        raise RuntimeError('baseline repository privacy has not been verified')
    print(json.dumps(dict(baseline_commit=head,harness_commit=git(harness,'rev-parse','HEAD'),private=True)))

if __name__=='__main__':
    main()
