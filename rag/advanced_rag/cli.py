"""Small commands for each stage; run the numbered entry points from project root."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
import pipeline as p

def main(stage):
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=f'Advanced RAG: {stage}')
    parser.add_argument('--debug', action='store_true')
    print(f'Advanced RAG: {stage} (Python {sys.version_info.major}.{sys.version_info.minor})')

    if stage == 'ingest':
        parser.add_argument('--chunking', choices=[*p.STRATEGIES,'all'], default='all')
        parser.add_argument('--chunk-size',type=int,default=700)
        parser.add_argument('--overlap',type=int,default=100)
        parser.add_argument('--batch-size',type=int,default=12)
        parser.add_argument('--force',action='store_true',help='Rebuild even when PDF, model, and settings are unchanged.')
    if stage in ('retrieve','answer','evaluate'):
        parser.add_argument('--chunking',choices=p.STRATEGIES,default='layout')
        parser.add_argument('--retrieval',choices=p.METHODS,default='hybrid')
        parser.add_argument('--compare',action='store_true',help='Run every chunking x retrieval combination.')
        parser.add_argument('--top-k',type=int,default=3)
        parser.add_argument('--candidates',type=int,default=20)
    if stage in ('retrieve','answer'):
        parser.add_argument('--question',default='How long does standard delivery take?')
    if stage == 'answer':
        parser.add_argument('--prompt-only',action='store_true')
        parser.add_argument('--context-chars',type=int,default=14000)
        parser.add_argument('--context-tokens',type=int,default=3000,help='Conservative evidence token budget (UTF-8 byte estimate).')
        parser.add_argument('--context-window',type=int,default=4096)
        parser.add_argument('--max-output-tokens',type=int,default=600)
    args = parser.parse_args()
    try:
        if stage == 'doctor':
            print('PDF:', p.PDF, 'exists:', p.PDF.exists())
            print('Embedding model:',p.MODEL,'digest:',p.model_digest())
            names = [m['name'] for m in p.ollama('tags')['models']]
            if p.LLM not in names:
                raise ValueError(f'Answer model missing: ollama pull {p.LLM}')
            with p.connect() as conn:
                p.init_db(conn)
                print('PostgreSQL + pgvector:',conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone())
            if not p.PDF.exists():
                raise ValueError('Create the handbook: python rag/advanced_rag/create_policy_pdf.py')
            print('Ready. Ingest if indexes have not been built.')
            return 0
        if stage == 'ingest':
            strategies = p.STRATEGIES if args.chunking == 'all' else [args.chunking]
            p.ingest(strategies,args.chunk_size,args.overlap,args.batch_size,args.force)
            return 0
        if args.top_k < 1 or args.candidates < args.top_k:
            parser.error('Use candidates >= top-k >= 1.')
        combos = [(s,m) for s in p.STRATEGIES for m in p.METHODS] if args.compare else [(args.chunking,args.retrieval)]
        digest = p.model_digest()
        with p.connect() as conn:
            for strategy in {s for s,_ in combos}:
                p.validate_run(conn,strategy,digest)
        if stage == 'evaluate':
            return evaluate(args,combos,digest)
        if not args.question.strip():
            parser.error('Question must not be empty.')
        if stage == 'answer' and min(args.context_chars,args.context_tokens,args.context_window,args.max_output_tokens) < 1:
            parser.error('Context/output budgets must be positive.')
        generating = stage == 'answer' and not args.prompt_only
        query_vector = p.embed([args.question],query=True,keep_alive=0 if generating else '5m')[0]
        if generating:
            print('Embedding model released before loading the answer model.',flush=True)
        report = []
        for strategy,method in combos:
            start = time.perf_counter()
            matches = p.retrieve(args.question,strategy,method,args.top_k,args.candidates,query_vector,digest)
            elapsed = time.perf_counter()-start
            print(f'\n=== {strategy} / {method} ===')
            for row in matches:
                print(f"{row['id']} | page {row['page']} | {row['kind']} | RRF {row['score']:.6f}")
                if 'sparse_score' in row or 'dense_score' in row:
                    print('BM25:',round(row['sparse_score'],4) if 'sparse_score' in row else 'outside sparse candidates',
                          '| Dense cosine:',round(row['dense_score'],4) if 'dense_score' in row else 'outside dense candidates')
                if row.get('matched_child'):
                    print('Matched child:',row['matched_child'])
                print(row['content'], '\n')
            item = {'chunking':strategy,'retrieval':method,'seconds':elapsed,'matches':matches,
                    'context_chars':sum(len(r['content']) for r in matches)}
            if stage == 'answer':
                prompt,used = p.build_prompt(args.question,matches,args.context_chars,args.context_tokens,
                                           args.context_window,args.max_output_tokens)
                print(f'Using {len(used)}/{len(matches)} results within context budget.')
                item['prompt_token_estimate_conservative'] = p.token_estimate(prompt)
                item['max_output_tokens'] = args.max_output_tokens
                print(f'Conservative prompt estimate: {p.token_estimate(prompt)}; output cap: {args.max_output_tokens}.')
                item['used_sources'] = [{'id':r['id'],'page':r['page']} for r in used]
                if args.prompt_only:
                    print(prompt)
                else:
                    result = p.ollama('generate',{'model':p.LLM,'prompt':prompt,'stream':False,'think':False,'keep_alive':0,
                        'options':{'temperature':0,'num_predict':args.max_output_tokens,'num_ctx':args.context_window}})
                    answer = result.get('response','').strip()
                    if not answer or result.get('done_reason') == 'length':
                        raise ValueError('Empty or truncated answer. Inspect context or increase generation limit.')
                    print('ANSWER:',answer)
                    item['answer'] = answer
                    item['llm_model'] = p.LLM
                    item['usage'] = {key:result.get(key) for key in ('prompt_eval_count','eval_count','total_duration')}
                    item['citation_check'] = p.check_citations(answer,used)
                    print('Actual Ollama usage:',item['usage'])
                    print('Citation check (IDs/pages only):',item['citation_check'])
            report.append(item)
        out = p.ROOT / 'results'
        out.mkdir(exist_ok=True)
        target = out / f'last_{stage}.json'
        target.write_text(json.dumps({'question':args.question,'model':p.MODEL,'model_digest':digest,
                                     'pdf_sha256':p.fingerprint(),'pipeline_version':p.VERSION,
                                     'top_k':args.top_k,'candidates':args.candidates,'results':report},indent=2),encoding='utf-8')
        print('Saved:',target)
        print('RRF is a ranking score, not confidence. Verify cited source text.')
        return 0
    except Exception as error:
        print(f'Could not {stage}: {error}\nCheck Docker, Ollama, dependencies, and run 00_check_setup.py.')
        if args.debug:
            traceback.print_exc()
        return 1

def evaluate(args, combos, digest):
    cases = json.loads((p.ROOT / 'eval_cases.json').read_text(encoding='utf-8'))
    totals = {f'{s}/{m}':{'hits':0,'reciprocal_rank':0.,'context_chars':0,'seconds':0.} for s,m in combos}
    details = []
    for number,case in enumerate(cases,1):
        print(f'Evaluating {number}/{len(cases)}: {case["question"]}',flush=True)
        vector = p.embed([case['question']],query=True)[0]
        for strategy,method in combos:
            start = time.perf_counter()
            rows = p.retrieve(case['question'],strategy,method,args.top_k,args.candidates,vector,digest)
            ranks = [i for i,r in enumerate(rows,1) if r['page'] in case['pages']]
            key = f'{strategy}/{method}'
            totals[key]['hits'] += bool(ranks)
            totals[key]['reciprocal_rank'] += 1/min(ranks) if ranks else 0
            totals[key]['context_chars'] += sum(len(r['content']) for r in rows)
            totals[key]['seconds'] += time.perf_counter()-start
            details.append({'question':case['question'],'configuration':key,'expected_pages':case['pages'],
                            'retrieved_pages':[r['page'] for r in rows]})
    summary = {key:{f'hit_at_{args.top_k}':v['hits']/len(cases), 'mrr':v['reciprocal_rank']/len(cases),
                    'mean_context_chars':v['context_chars']/len(cases),
                    'mean_retrieval_seconds_excluding_query_embedding':v['seconds']/len(cases)} for key,v in totals.items()}
    target = p.ROOT / 'results/evaluation.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps({'cases':len(cases),'model':p.MODEL,'model_digest':digest,
                                 'pdf_sha256':p.fingerprint(),'pipeline_version':p.VERSION,
                                 'top_k':args.top_k,'candidates':args.candidates,
                                 'summary':summary,'details':details},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('Saved:',target)
    print('Page-level retrieval checks only; not a measure of answer correctness or refusal quality.')
    return 0
