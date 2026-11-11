import unittest
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pipeline as p

class PipelineTests(unittest.TestCase):
    def test_answer_releases_models_and_saves_actual_token_usage(self):
        import cli
        rows=[{'id':'c1','page':1,'kind':'section','score':.03,'content':'Returns within 30 days.'}]
        response={'response':'30 days [Source c1 | PDF page 1].','prompt_eval_count':123,'eval_count':20}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder, patch.object(p,'ROOT',Path(folder)), \
             patch.object(p,'model_digest',return_value='digest'), patch.object(p,'connect'), \
             patch.object(p,'validate_run'), patch.object(p,'retrieve',return_value=rows), \
             patch.object(p,'embed',return_value=[[.1]*768]) as embed, \
             patch.object(p,'ollama',return_value=response) as generate, patch.object(sys,'stdout'), \
             patch.object(sys,'argv',['answer','--question','When can I return?']):
            self.assertEqual(cli.main('answer'),0)
            self.assertEqual(embed.call_args.kwargs['keep_alive'],0)
            self.assertEqual(generate.call_args.args[1]['keep_alive'],0)
            self.assertEqual(generate.call_args.args[1]['options']['num_ctx'],4096)
            report=json.loads((Path(folder)/'results/last_answer.json').read_text())
            self.assertEqual(report['results'][0]['usage']['prompt_eval_count'],123)
            self.assertTrue(report['results'][0]['citation_check']['has_valid_citation'])

    def test_pdf_has_100_pages_tables_and_10_illustrated_pages(self):
        from pypdf import PdfReader
        reader = PdfReader(p.PDF)
        self.assertEqual(len(reader.pages),100)
        self.assertEqual(sum(bool(page.images) for page in reader.pages),10)
        self.assertTrue(all('Reference table' in page.extract_text() for page in reader.pages))

    def test_real_layout_extraction_keeps_column_labels_and_captions(self):
        pages = p.extract_pages()
        self.assertEqual(len(pages),100)
        self.assertTrue(all('(cid:127)' not in page['content'] for page in pages))
        self.assertTrue(all(any(b['kind']=='table_row' for b in page['blocks']) for page in pages))
        self.assertEqual(sum(any(b['kind']=='figure_caption' for b in page['blocks']) for page in pages),10)
        self.assertTrue(any('Policy requirement:' in b['text'] for b in pages[0]['blocks']))
        for strategy in p.STRATEGIES:
            chunks = p.make_chunks(pages,strategy)
            self.assertTrue(all(len(c['content']) <= 700 for c in chunks))
            self.assertEqual({c['page'] for c in chunks},set(range(1,101)))

    def test_context_budget_deduplicates_and_reserves_output(self):
        rows = [{'id':'c1','page':1,'content':'A policy'}, {'id':'c2','page':2,'content':'A  policy'}]
        _,used = p.build_prompt('what?',rows)
        self.assertEqual(len(used),1)
        with self.assertRaisesRegex(ValueError,'no room'):
            p.build_prompt('q'*1000,rows,context_window=1024,max_output_tokens=600)
        with self.assertRaisesRegex(ValueError,'No evidence fits'):
            p.build_prompt('what?',rows,context_tokens=10)

    def test_citations_check_both_id_and_page(self):
        rows=[{'id':'c1','page':2}]
        check=p.check_citations('Yes [Source c1 | PDF page 2]. No [Source c1 | PDF page 9].',rows)
        self.assertTrue(check['has_valid_citation'])
        self.assertEqual(check['invalid_citations'],[('c1',9)])
        self.assertFalse(p.check_citations('No references',rows)['has_valid_citation'])

    def test_unchanged_ingest_skips_embedding_and_extraction(self):
        conn=MagicMock()
        conn.execute.return_value.fetchone.return_value={'fingerprint':'same','model':p.MODEL,'digest':'same',
            'config':{'size':700,'overlap':100,'version':p.VERSION,'dimensions':768}}
        with patch.object(p,'connect') as connect, patch.object(p,'fingerprint',return_value='same'), \
             patch.object(p,'model_digest',return_value='same'), patch.object(p,'extract_pages') as extract, \
             patch.object(p,'embed') as embed:
            connect.return_value.__enter__.return_value=conn
            p.ingest()
            extract.assert_not_called()
            embed.assert_not_called()

    def test_parent_retrieval_deduplicates_before_top_k(self):
        children = [dict(id='c1',parent_id='p1',page=1,title='A',kind='section',content='refund'),
                    dict(id='c2',parent_id='p1',page=1,title='A',kind='section',content='refund'),
                    dict(id='c3',parent_id='p2',page=2,title='B',kind='section',content='refund')]
        conn = MagicMock()
        def execute(sql, params):
            result = MagicMock()
            if 'FROM advanced_parents' in sql:
                result.fetchone.return_value = {'id':params[1],'page':int(params[1][-1]),'content':'complete parent'}
            else:
                result.fetchall.return_value = children
            return result
        conn.execute.side_effect = execute
        with patch.object(p,'connect') as connect, patch.object(p,'validate_run'):
            connect.return_value.__enter__.return_value = conn
            rows = p.retrieve('refund',method='parent_child',top_k=2,candidates=3,query_vector=[.1]*768,digest='digest')
        self.assertEqual([r['id'] for r in rows],['p1','p2'])
        self.assertEqual(rows[0]['matched_child'],'c1')

    def test_stale_model_digest_is_rejected(self):
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = {'fingerprint':'same','model':p.MODEL,
            'digest':'old','config':{'version':p.VERSION}}
        with patch.object(p,'fingerprint',return_value='same'):
            with self.assertRaisesRegex(ValueError,'stale'):
                p.validate_run(conn,'layout','new')

    def test_invalid_embedding_dimensions_fail(self):
        with patch.object(p,'ollama',return_value={'embeddings':[[1,2]]}):
            with self.assertRaisesRegex(ValueError,'768'):
                p.embed(['text'])

    def test_bm25_rare_term_and_no_match(self):
        rows = [{'id':'a','content':'standard policy ZX-99'}, {'id':'b','content':'standard policy'},
                {'id':'c','content':'standard policy delivery'}]
        self.assertEqual(p.bm25('ZX-99',rows)[0]['id'],'a')
        self.assertEqual(p.bm25('absent',rows),[])

    def test_rrf_rewards_agreement_without_mixing_raw_scores(self):
        rows = p.fuse([{'id':'a','sparse_score':1000},{'id':'b','sparse_score':1}],
                      [{'id':'b','dense_score':.8},{'id':'c','dense_score':.7}])
        self.assertEqual(rows[0]['id'],'b')
        self.assertAlmostEqual(rows[0]['score'],1/62+1/61)

    def test_layout_keeps_rows_and_parent_links(self):
        pages = [{'id':'p1','page':1,'title':'Delivery','content':'plain text',
                  'blocks':[{'text':'Field: Express | Days: 2','kind':'table_row'}]}]
        rows = p.make_chunks(pages,'layout')
        self.assertEqual(rows[0]['parent_id'],'p1')
        self.assertIn('Days: 2',rows[0]['content'])
        self.assertEqual(rows[0]['kind'],'table_row')
        self.assertEqual(p.make_chunks(pages,'recursive')[0]['content'],'plain text')

    def test_invalid_chunk_settings(self):
        for size, overlap in [(0,0),(10,10),(10,-1)]:
            with self.assertRaises(ValueError):
                p.make_chunks([],'layout',size,overlap)

    def test_context_budget_preserves_whole_sources(self):
        rows=[{'id':'c1','page':2,'content':'hello'}, {'id':'c2','page':3,'content':'x'*1000}]
        prompt,used=p.build_prompt('what?',rows,budget=100)
        self.assertEqual(len(used),1)
        self.assertIn('PDF page 2',prompt)
        self.assertNotIn('PDF page 3',prompt)
        with self.assertRaises(ValueError):
            p.build_prompt('what?',rows,budget=1)

    def test_embedding_task_prefixes(self):
        vector=[.1]*768
        with patch.object(p,'MODEL','embeddinggemma:latest'), patch.object(p,'ollama',return_value={'embeddings':[vector]}) as call:
            p.embed(['question'],query=True)
            self.assertTrue(call.call_args.args[1]['input'][0].startswith('task: search result | query:'))
            p.embed(['document'])
            self.assertTrue(call.call_args.args[1]['input'][0].startswith('title:'))

if __name__ == '__main__':
    unittest.main()
