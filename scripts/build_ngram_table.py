#!/usr/bin/env python3
"""Count response token IDs. Smoothing is applied by NGramGPU at query time."""
import argparse,json,time
from pathlib import Path
import numpy as np
from dforktree.counts import Counter3,context_sums

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True,help='JSONL with a resp_ids list per response');p.add_argument('--out',default='data/ngram_counts.npz');p.add_argument('--vocab-size',type=int,default=248077);p.add_argument('--min3',type=int,default=2);p.add_argument('--chunk-tokens',type=int,default=20_000_000);a=p.parse_args()
 t=time.perf_counter();c=Counter3(a.vocab_size,a.chunk_tokens);rows=0
 with open(a.data,encoding='utf-8') as f:
  for line in f:
   if not line.strip():continue
   x=json.loads(line);c.add(x['resp_ids']);rows+=1
 k2,c2,k3,c3,dropped=c.finish(a.min3);ctx,tot=context_sums(k3,c3,a.vocab_size)
 path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
 np.savez(path,keys=k2,cab=c2.astype(np.float32),prev=c.prev.astype(np.float32),nxt=c.nxt,N=c.N,V=a.vocab_size,keys3=k3,c3=c3.astype(np.float32),keys3c=ctx,n3c=tot,min3=a.min3,with_prompt=False)
 print(json.dumps(dict(responses=rows,tokens=c.n_tok,bigram_types=len(k2),trigram_types=len(k3),trigram_types_dropped=dropped,seconds=time.perf_counter()-t)))
if __name__=='__main__':main()
