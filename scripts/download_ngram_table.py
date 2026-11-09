"""Download the table from this repository's release assets, outside Git history."""
from pathlib import Path
import argparse,json,os,urllib.request,shutil
p=argparse.ArgumentParser();p.add_argument('--tag',default='v0.1.0');p.add_argument('--output-dir',default='data');a=p.parse_args()
headers={'User-Agent':'DForkTree'}
token=os.environ.get('GITHUB_TOKEN')
if token:headers['Authorization']='Bearer '+token
api='https://api.github.com/repos/Perseus037/DForkTree/releases/tags/'+a.tag
with urllib.request.urlopen(urllib.request.Request(api,headers=headers)) as f:release=json.load(f)
asset=next(x for x in release['assets'] if x['name']=='ngram_counts.npz')
if token:
 url=asset['url'];headers['Accept']='application/octet-stream'
else:url=asset['browser_download_url']
dest=Path(a.output_dir)/'ngram_counts.npz';dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.part')
with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as src,tmp.open('wb') as out:shutil.copyfileobj(src,out)
if tmp.stat().st_size!=asset['size']:raise RuntimeError('Incomplete download; rerun the command.')
tmp.replace(dest);print(dest)
