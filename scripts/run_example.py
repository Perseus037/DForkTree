"""Small batch-one T=0 example with an optional native AR speed reference."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,urllib.request,urllib.error,signal,tempfile

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['4b','9b'],default='4b');p.add_argument('--target');p.add_argument('--draft');p.add_argument('--ngram-table',default='data/ngram_counts.npz');p.add_argument('--port',type=int,default=30484);p.add_argument('--max-new-tokens',type=int,default=256);p.add_argument('--skip-ar',action='store_true');p.add_argument('--output',default='example_results.json');a=p.parse_args()
 table=Path(a.ngram_table).resolve()
 if not table.is_file():p.error('N-gram table not found. Download it first or pass --ngram-table.')
 from transformers import AutoTokenizer
 target=a.target or 'Qwen/Qwen3.5-'+a.model.upper();draft=a.draft or 'z-lab/Qwen3.5-'+a.model.upper()+'-DFlash'
 tokenizer=AutoTokenizer.from_pretrained(target,trust_remote_code=True)
 examples=[json.loads(line) for line in (Path(__file__).resolve().parents[1]/'examples/gsm8k_small.jsonl').read_text().splitlines()]
 prompts=[tokenizer.apply_chat_template([dict(role='user',content=x['question'])],tokenize=True,add_generation_prompt=True,enable_thinking=False) for x in examples]
 base=f'http://127.0.0.1:{a.port}';results={}
 def request(endpoint,body=None,timeout=600):
  req=urllib.request.Request(base+endpoint,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=timeout) as f:return True if endpoint=='/health' else json.load(f)
 for mode in (['dforktree'] if a.skip_ar else ['ar','dforktree']):
  env=os.environ.copy()
  for key in list(env):
   if key.startswith('FC1_') or key=='SGLANG_PLUGINS':env.pop(key)
  cmd=[sys.executable,'-m','sglang.launch_server','--model-path',target,'--trust-remote-code','--dtype','bfloat16','--host','127.0.0.1','--port',str(a.port),'--context-length','8192','--max-running-requests','1','--max-mamba-cache-size','8' if a.model=='4b' else '5','--max-total-tokens','8192','--mem-fraction-static','.70' if a.model=='4b' else '.90','--cuda-graph-max-bs-decode','1','--random-seed','20261103']
  if a.model=='9b':cmd.append('--disable-prefill-cuda-graph')
  if mode=='dforktree':
   env.update(SGLANG_PLUGINS='dforktree',FC1_TREE64='1',FC1_BRANCH_PILOT='1',FC1_FUSED='1',FC1_OURS='1',FC1_COMPILE_BUILD='1',FC1_TREE_N='32',FC1_DRAFT_BLOCK='24',FC1_CHAIN_FIRST='0',FC1_NGRAM_PATH=str(table))
   cmd+=['--speculative-algorithm','FC1_DFLASH_TREE','--speculative-draft-model-path',draft,'--speculative-dflash-block-size','24','--speculative-accept-threshold-single','1.0','--speculative-accept-threshold-acc','1.0']
  log=tempfile.NamedTemporaryFile(prefix='dforktree-'+mode+'-',suffix='.log',delete=False);log.close()
  print(f'Starting {mode}; server log: {log.name}',flush=True)
  with open(log.name,'w') as stream:proc=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   ready=False
   for _ in range(240):
    if proc.poll() is not None:raise RuntimeError(f'{mode} exited {proc.returncode}; see {log.name}')
    try:request('/health',timeout=2);ready=True;break
    except (OSError,ValueError):time.sleep(2)
   if not ready:raise RuntimeError('Server did not become ready; see '+log.name)
   def body(ids,n):return dict(input_ids=ids,sampling_params=dict(temperature=0.,top_p=1.,top_k=-1,min_p=0.,sampling_seed=20261103,max_new_tokens=n),stream=False)
   request('/generate',body(prompts[0],64));request('/generate',body(prompts[0],128))
   rows=[]
   for example,ids in zip(examples,prompts):
    request('/flush_cache');start=time.perf_counter();out=request('/generate',body(ids,a.max_new_tokens));elapsed=time.perf_counter()-start
    meta=out['meta_info'];tokens=int(meta['completion_tokens']);verify=int(meta['spec_verify_ct']) if mode=='dforktree' else None
    if mode=='dforktree' and verify<=0:raise RuntimeError('Missing verification rounds; refusing to report a fabricated tau')
    rows.append(dict(id=example['id'],output_tokens=tokens,wall_seconds=elapsed,verify_rounds=verify,finish_reason=meta['finish_reason']))
   tokens=sum(r['output_tokens'] for r in rows);seconds=sum(r['wall_seconds'] for r in rows)
   results[mode]=dict(prompts=rows,tokens_per_second=tokens/seconds,tau=tokens/sum(r['verify_rounds'] for r in rows) if mode=='dforktree' else None)
   print(json.dumps({mode:results[mode]},indent=2),flush=True)
  finally:
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 if 'ar' in results:results['speedup_vs_ar']=results['dforktree']['tokens_per_second']/results['ar']['tokens_per_second']
 Path(a.output).write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
if __name__=='__main__':main()

