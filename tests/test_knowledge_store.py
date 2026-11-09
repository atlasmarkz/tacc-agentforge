"""Focused store regressions, all synthetic/offline."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from knowledge.store import (audit_database, export_snapshot, ingest_runs, verify_snapshot,
                             _group_reviews, currency_for, _resolve_json_pointer)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / 'knowledge/catalog.sqlite'
        self.config = self.root / 'queries.json'
        self.config.write_text(json.dumps({'queries': [{'q': 'test packs', 'category': 'backpacks'}]}))
        self.schemas = Path(__file__).resolve().parents[1] / 'configs/category_schemas.json'

    def tearDown(self):
        self.tmp.cleanup()

    def run_archive(self, name, responses):
        run = self.root / name
        (run / 'raw').mkdir(parents=True)
        lines = []
        for i, (engine, body, when, meta) in enumerate(responses):
            data = json.dumps(body, sort_keys=True).encode()
            path = f'raw/{i}.json'
            (run / path).write_bytes(data)
            params = {'q': 'test packs', 'gl': 'us', 'hl': 'en'} if engine == 'google_shopping' else {'page_token': 'token', 'gl': 'us', 'hl': 'en'}
            record = {'call_id': f'{name}-{i}', 'engine': engine, 'status': 'ok', 'params': params,
                      'raw_path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'requested_at': when}
            record.update(meta)
            lines.append(json.dumps(record))
        (run / 'index.jsonl').write_text('\n'.join(lines) + '\n')
        return run

    def listing(self, pid='p1', token='token'):
        return {'shopping_results': [{'product_id': pid, 'immersive_product_page_token': token,
                'title': 'Travel pack', 'price': '$10.00', 'extracted_price': 10,
                'product_link': 'https://example.com/p1', 'source': 'Shop'}]}

    def detail(self, price='USD 10.00', shipping='Free shipping', stores=None):
        return {'product_results': {'title': 'Travel pack', 'price_insights': {'currency': 'USD'},
                'product_attributes': [{'name': 'Capacity', 'value': '70 L'},
                                       {'name': 'Waterproof', 'value': 'Yes'},
                                       {'name': 'Material', 'value': 'Nylon'}],
                'stores': stores if stores is not None else [{'name': 'Shop', 'title': 'Travel pack 70L',
                            'link': 'https://merchant.example/pack', 'price': price,
                            'extracted_price': float(price.split()[-1]), 'shipping': shipping}]}}

    def import_(self, *runs):
        return ingest_runs(list(runs), self.db, self.config, self.schemas)

    def rows(self, query):
        with sqlite3.connect(self.db) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(query)]

    def test_idempotent_import_and_fts_rowids(self):
        run = self.run_archive('one', [('google_shopping', self.listing(), '2026-01-01T00:00:00Z', {}),
                ('google_product', self.detail(), '2026-01-01T00:01:00Z', {})])
        first = self.import_(run)
        timestamps = self.rows('SELECT evidence_id,observed_at FROM evidence ORDER BY id')
        second = self.import_(run)
        self.assertEqual(first['counts'], second['counts'])
        self.assertEqual(second['imported_raw_records'], 0)
        self.assertEqual(timestamps, self.rows('SELECT evidence_id,observed_at FROM evidence ORDER BY id'))
        self.assertEqual(audit_database(self.db)['fts_rowid_errors'], 0)
        self.assertEqual(self.rows('SELECT e.id FROM evidence e JOIN evidence_fts f ON e.id=f.rowid WHERE e.evidence_id<>f.evidence_id'), [])

    def test_multirun_source_history_uses_latest_not_historical_minimum(self):
        listing = self.run_archive('listing', [('google_shopping', self.listing(), '2026-01-01T00:00:00Z', {})])
        old = self.run_archive('old', [('google_product', self.detail('USD 5.00'), '2026-01-01T01:00:00Z', {})])
        new = self.run_archive('new', [('google_product', self.detail('USD 20.00'), '2026-01-02T01:00:00Z', {})])
        self.import_(listing, new)
        self.import_(old)
        self.assertEqual(self.rows('SELECT COUNT(*) n FROM offer_observations')[0]['n'], 2)
        current = self.rows('SELECT price_minor,cost_conflict FROM current_offer_observations')
        self.assertEqual(current, [{'price_minor': 2000, 'cost_conflict': 0}])
        self.assertEqual(self.rows('SELECT COUNT(*) n FROM variants')[0]['n'], 1)

    def test_same_timestamp_cost_conflict_remains_unselected(self):
        run = self.run_archive('conflict', [('google_shopping', self.listing(), '2026-01-01T00:00:00Z', {}),
            ('google_product', self.detail('USD 5.00'), '2026-01-01T01:00:00Z', {}),
            ('google_product', self.detail('USD 9.00'), '2026-01-01T01:00:00Z', {})])
        result = self.import_(run)
        self.assertEqual(result['current_cost_conflicts'], 1)
        self.assertEqual({r['price_minor'] for r in self.rows('SELECT * FROM current_offer_observations')}, {500, 900})
        self.assertTrue(all(r['cost_conflict'] == 1 for r in self.rows('SELECT * FROM current_offer_observations')))

    def test_same_url_different_capacity_offers_stay_separate(self):
        stores = [{'name': 'Shop', 'title': f'Travel pack {capacity}L', 'link': 'https://merchant.example/pack',
                   'price': 'USD 10.00', 'extracted_price': 10, 'selected_options': {'capacity': f'{capacity}L'}} for capacity in (143, 70)]
        run = self.run_archive('capacity', [('google_shopping', self.listing(), '2026-01-01T00:00:00Z', {}),
            ('google_product', self.detail(stores=stores), '2026-01-01T01:00:00Z', {})])
        self.import_(run)
        self.assertEqual(len(self.rows('SELECT * FROM offers')), 2)
        self.assertTrue(all(r['identity_status'] == 'provider_group_unverified' for r in self.rows('SELECT * FROM offers')))

    def test_explicit_crossrun_metadata_overrides_ambiguous_token(self):
        listings = self.listing('p1')
        listings['shopping_results'] += self.listing('p2')['shopping_results']
        one = self.run_archive('ambiguous', [('google_shopping', listings, '2026-01-01T00:00:00Z', {})])
        two = self.run_archive('explicit', [('google_product', self.detail(), '2026-01-02T00:00:00Z', {'product_id': 'p2'})])
        self.import_(one)
        self.import_(two)
        self.assertEqual(self.rows("SELECT provider_product_id FROM variants WHERE detail_status='details_available'"), [{'provider_product_id': 'p2'}])

    def test_strict_currency_conditional_shipping_and_unknown_tax(self):
        detail = self.detail()
        detail['product_results'].pop('price_insights')
        detail['product_results']['stores'] = [
            {'name': 'Bare', 'title': 'Bare', 'price': '$10.00', 'extracted_price': 10, 'shipping': '$2.00'},
            {'name': 'Cross', 'title': 'Cross', 'price': 'USD 10.00', 'extracted_price': 10, 'shipping': 'HK$2.00', 'shipping_extracted': 2},
            {'name': 'Conditional', 'title': 'Conditional', 'price': 'USD 10.00', 'extracted_price': 10, 'shipping': 'Free shipping on orders over $50'},
            {'name': 'Prime', 'title': 'Prime', 'price': 'USD 10.00', 'extracted_price': 10, 'shipping': 'Free shipping', 'details_and_offers': ['Prime members only']},
        ]
        run = self.run_archive('fees', [('google_shopping', self.listing(), '2026-01-01T00:00:00Z', {}),
                                      ('google_product', detail, '2026-01-01T01:00:00Z', {})])
        self.import_(run)
        rows = self.rows('SELECT o.merchant,x.* FROM current_offer_observations x JOIN offers o USING(offer_id)')
        self.assertIsNone(next(r for r in rows if r['merchant'] == 'Bare')['currency'])
        self.assertTrue(all(r['shipping_status'] == 'unknown' and r['shipping_minor'] is None for r in rows))
        self.assertTrue(all(r['tax_status'] == 'unknown' and r['tax_minor'] is None for r in rows))

    def test_typed_attributes_and_unknown_values(self):
        detail = self.detail()
        detail['product_results']['product_attributes'].append({'name': 'Laptop size', 'value': 'up to 16 inches'})
        run = self.run_archive('attrs', [('google_shopping', self.listing(), '2026-01-01T00:00:00Z', {}),
                                       ('google_product', detail, '2026-01-01T01:00:00Z', {})])
        self.import_(run)
        rows = {r['canonical_name']: r for r in self.rows('SELECT * FROM current_attributes')}
        self.assertEqual(rows['capacity']['normalized_number'], '70')
        self.assertEqual(rows['capacity']['numeric_value'], 70.0)
        self.assertEqual(rows['capacity']['unit'], 'L')
        self.assertEqual(rows['water_resistant']['bool_value'], 1)
        self.assertEqual(rows['material']['text_value'], 'nylon')
        self.assertEqual(rows['laptop_size']['status'], 'unknown')

    def test_snapshot_portability_integrity_and_immutability(self):
        run = self.run_archive('snapshot', [('google_shopping', self.listing(), '2026-01-01T00:00:00Z', {}),
                     ('google_product', self.detail(), '2026-01-01T01:00:00Z', {})])
        self.import_(run)
        first = export_snapshot(self.db, self.root / 'snapshots')
        snapshot = Path(first['path'])
        before = (snapshot / 'catalog.sqlite').read_bytes()
        self.assertTrue(verify_snapshot(snapshot)['valid'])
        self.assertEqual(export_snapshot(self.db, self.root / 'snapshots')['snapshot_id'], first['snapshot_id'])
        later = self.run_archive('later', [('google_product', self.detail('USD 50.00'), '2026-01-02T01:00:00Z', {})])
        self.import_(later)
        self.assertEqual(before, (snapshot / 'catalog.sqlite').read_bytes())
        self.assertTrue(verify_snapshot(snapshot)['valid'])
        import shutil
        moved = self.root / 'portable'
        shutil.copytree(snapshot, moved)
        self.assertTrue(verify_snapshot(moved)['valid'])
        raw = next((moved / 'raw').iterdir())
        raw.write_text('{}')
        self.assertFalse(verify_snapshot(moved)['valid'])

    def test_archive_hash_and_path_rejected(self):
        run = self.run_archive('bad', [('google_shopping', self.listing(), '2026-01-01T00:00:00Z', {})])
        (run / 'raw/0.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.import_(run)
        self.assertFalse(self.db.exists())

    def test_near_duplicate_review_groups_keep_split_and_prevent_leakage(self):
        body = self.detail()
        text = 'This bag has a very comfortable shoulder strap and plenty of useful pockets for daily travel'
        body['product_results']['user_reviews'] = [{'text': text, 'rating': 5}]
        first = self.run_archive('reviews1', [('google_shopping', self.listing(), '2026-01-01T00:00:00Z', {}),
            ('google_product', body, '2026-01-01T01:00:00Z', {})])
        self.import_(first)
        group = self.rows('SELECT duplicate_group_id FROM reviews')[0]['duplicate_group_id']
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE reviews SET split='held_out'")
            conn.execute("UPDATE review_duplicate_groups SET split='held_out'")
        second_body = deepcopy(body)
        second_body['product_results']['user_reviews'][0]['title'] = 'Helpful'
        second = self.run_archive('reviews2', [('google_product', second_body, '2026-01-02T01:00:00Z', {})])
        self.import_(second)
        self.assertEqual(self.rows('SELECT DISTINCT duplicate_group_id,split FROM reviews'), [{'duplicate_group_id': group, 'split': 'held_out'}])

    def test_archive_path_escape_rejected_before_database_creation(self):
        run = self.run_archive('escape', [('google_shopping', self.listing(), '2026-01-01T00:00:00Z', {})])
        record = json.loads((run / 'index.jsonl').read_text())
        record['raw_path'] = '../outside.json'
        (run / 'index.jsonl').write_text(json.dumps(record) + '\n')
        with self.assertRaisesRegex(ValueError, 'safe relative'):
            self.import_(run)
        self.assertFalse(self.db.exists())

    def test_source_secrets_sanitized_without_reading_credentials(self):
        body = self.listing()
        body['api_key'] = 'synthetic-secret-value'
        body['nested'] = {'access_token': 'synthetic-secret-value', 'url': 'https://example.com?q=x&api_key=synthetic-secret-value'}
        run = self.run_archive('sanitize', [('google_shopping', body, '2026-01-01T00:00:00Z', {})])
        self.import_(run)
        for path in (self.db.parent / 'raw').glob('*.json'):
            self.assertNotIn('synthetic-secret-value', path.read_text())
        exported = export_snapshot(self.db, self.root / 'snapshots')
        self.assertTrue(verify_snapshot(Path(exported['path']))['valid'])

    def test_reviewer_split_conflict_is_rejected(self):
        body = self.detail()
        text = 'This bag has a very comfortable shoulder strap and plenty of useful pockets for daily travel'
        body['product_results']['user_reviews'] = [{'text': text}, {'text': text}]
        run = self.run_archive('splitconflict', [('google_shopping', self.listing(), '2026-01-01T00:00:00Z', {}),
                                               ('google_product', body, '2026-01-01T01:00:00Z', {})])
        self.import_(run)
        with sqlite3.connect(self.db) as conn:
            conn.row_factory = sqlite3.Row
            ids = [row[0] for row in conn.execute('SELECT review_id FROM reviews ORDER BY review_id')]
            conn.execute("UPDATE reviews SET split='development' WHERE review_id=?", (ids[0],))
            conn.execute("UPDATE reviews SET split='held_out' WHERE review_id=?", (ids[1],))
            with self.assertRaisesRegex(ValueError, 'cross data splits'):
                _group_reviews(conn)

    def test_schema_change_retypes_archived_facts_without_new_evidence(self):
        body = self.detail()
        body['product_results']['product_attributes'].append({'name': 'New field', 'value': '42 L'})
        run = self.run_archive('schema', [('google_shopping', self.listing(), '2026-01-01T00:00:00Z', {}),
                                        ('google_product', body, '2026-01-01T01:00:00Z', {})])
        self.import_(run)
        count = self.rows('SELECT COUNT(*) n FROM evidence')[0]['n']
        config = json.loads(self.schemas.read_text())
        config['categories']['backpacks']['attributes']['new_field'] = {
            'type': 'number', 'unit': 'L', 'aliases': ['New field'], 'units': {'L': '1'}}
        custom = self.root / 'category_schemas.json'
        custom.write_text(json.dumps(config))
        ingest_runs([run], self.db, self.config, custom)
        self.assertEqual(self.rows('SELECT COUNT(*) n FROM evidence')[0]['n'], count)
        self.assertEqual(self.rows("SELECT normalized_number,status FROM current_attributes WHERE canonical_name='new_field'"),
                         [{'normalized_number': '42', 'status': 'known'}])

    def test_explicit_currency_symbols_do_not_match_embedded_suffixes(self):
        self.assertEqual(currency_for('US$10.00')[0], 'USD')
        self.assertEqual(currency_for('CA$10.00')[0], 'CAD')
        self.assertEqual(currency_for('S$10.00')[0], 'SGD')
        self.assertIsNone(currency_for('$10.00')[0])
        self.assertIsNone(currency_for('USD 10 / HK$78')[0])

    def test_rfc6901_pointer_dictionary_escapes_and_array_indexes(self):
        body = {'a/b': {'~name': [None, {'~1': 42}]}, '01': 'dictionary key', '': 'empty key'}
        self.assertIs(_resolve_json_pointer(body, ''), body)
        self.assertEqual(_resolve_json_pointer(body, '/a~1b/~0name/1/~01'), 42)
        self.assertIsNone(_resolve_json_pointer(body, '/a~1b/~0name/0'))
        self.assertEqual(_resolve_json_pointer(body, '/01'), 'dictionary key')
        self.assertEqual(_resolve_json_pointer(body, '/'), 'empty key')
        for invalid in ('a~1b', '/a~2b', '/missing', '/a~1b/~0name/-', '/a~1b/~0name/01',
                        '/a~1b/~0name/-1', '/a~1b/~0name/2', '/01/scalar'):
            with self.subTest(pointer=invalid):
                with self.assertRaises(ValueError):
                    _resolve_json_pointer(body, invalid)

    def test_snapshot_verifier_checks_every_evidence_pointer_even_with_updated_file_hash(self):
        run = self.run_archive('pointers', [('google_shopping', self.listing(), '2026-01-01T00:00:00Z', {}),
                  ('google_product', self.detail(), '2026-01-01T01:00:00Z', {})])
        self.import_(run)
        exported = export_snapshot(self.db, self.root / 'snapshots')
        snapshot = Path(exported['path'])
        count = exported['counts']['evidence']
        good = verify_snapshot(snapshot)
        self.assertTrue(good['valid'])
        self.assertEqual(good['evidence_pointers_checked'], count)
        # Re-hashing a deliberately damaged database must not make its pointers
        # valid. The logical fingerprint will also fail; assert the pointer error.
        with sqlite3.connect(snapshot / 'catalog.sqlite') as conn:
            conn.execute("UPDATE evidence SET json_pointer='/shopping_results/999' WHERE id=(SELECT MIN(id) FROM evidence)")
        manifest_path = snapshot / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files']['catalog.sqlite'] = hashlib.sha256((snapshot / 'catalog.sqlite').read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        damaged = verify_snapshot(snapshot)
        self.assertFalse(damaged['valid'])
        self.assertEqual(damaged['evidence_pointers_checked'], count)
        self.assertTrue(any('Evidence pointer invalid' in error for error in damaged['errors']))


if __name__ == '__main__':
    unittest.main()
