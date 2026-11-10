PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS category_schemas (category_id TEXT PRIMARY KEY, schema_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS raw_records (raw_id TEXT PRIMARY KEY,call_id TEXT NOT NULL,engine TEXT NOT NULL,origin_sha256 TEXT NOT NULL,archive_sha256 TEXT NOT NULL,archive_path TEXT NOT NULL,observed_at TEXT NOT NULL,processed_at TEXT,params_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS imports (import_id TEXT PRIMARY KEY,index_sha256 TEXT NOT NULL,config_sha256 TEXT NOT NULL,provenance_json TEXT NOT NULL,imported_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS import_raw_records (import_id TEXT NOT NULL REFERENCES imports,raw_id TEXT NOT NULL REFERENCES raw_records,PRIMARY KEY(import_id,raw_id));
CREATE TABLE IF NOT EXISTS variants (variant_id TEXT PRIMARY KEY,provider_product_id TEXT,title TEXT NOT NULL,category TEXT NOT NULL,type TEXT NOT NULL DEFAULT 'unknown',identity_status TEXT NOT NULL,detail_status TEXT NOT NULL,source_url TEXT,currency TEXT,currency_basis TEXT NOT NULL,observed_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS listings (listing_id TEXT PRIMARY KEY,variant_id TEXT NOT NULL REFERENCES variants,raw_id TEXT NOT NULL REFERENCES raw_records,json_pointer TEXT NOT NULL,query TEXT NOT NULL,category_hint TEXT NOT NULL,source_url TEXT,merchant TEXT,title TEXT NOT NULL,price_minor INTEGER CHECK(price_minor >= 0),currency TEXT,delivery_text TEXT,observed_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS offers (offer_id TEXT PRIMARY KEY,variant_id TEXT NOT NULL REFERENCES variants,merchant_id TEXT,merchant TEXT NOT NULL,url TEXT,identity_status TEXT NOT NULL,store_title TEXT NOT NULL,selected_options_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence (id INTEGER PRIMARY KEY,evidence_id TEXT UNIQUE NOT NULL,variant_id TEXT NOT NULL REFERENCES variants,kind TEXT NOT NULL,title TEXT NOT NULL,text TEXT NOT NULL,source_url TEXT,source_label TEXT,raw_id TEXT NOT NULL REFERENCES raw_records,json_pointer TEXT NOT NULL,observed_at TEXT NOT NULL,scope TEXT NOT NULL,duplicate_group_id TEXT);
CREATE TABLE IF NOT EXISTS offer_observations (
 observation_id TEXT PRIMARY KEY,offer_id TEXT NOT NULL REFERENCES offers,raw_id TEXT NOT NULL REFERENCES raw_records,evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),observed_at TEXT NOT NULL,currency TEXT,currency_basis TEXT NOT NULL,price_minor INTEGER CHECK(price_minor >= 0),shipping_minor INTEGER CHECK(shipping_minor >= 0),shipping_status TEXT NOT NULL CHECK(shipping_status IN ('known','unknown')),shipping_scope TEXT NOT NULL,tax_minor INTEGER CHECK(tax_minor >= 0),tax_status TEXT NOT NULL CHECK(tax_status IN ('known','unknown')),price_includes_tax TEXT NOT NULL,shipping_text TEXT,additional_costs TEXT,displayed_total_minor INTEGER,context_json TEXT NOT NULL,context_key TEXT NOT NULL,cost_signature TEXT NOT NULL,condition TEXT,store_title TEXT,
 CHECK((shipping_status='unknown' AND shipping_minor IS NULL) OR (shipping_status='known' AND shipping_minor IS NOT NULL)),CHECK((tax_status='unknown' AND tax_minor IS NULL) OR (tax_status='known' AND tax_minor IS NOT NULL)));
CREATE TABLE IF NOT EXISTS attributes (attribute_id TEXT PRIMARY KEY,variant_id TEXT NOT NULL REFERENCES variants,name TEXT NOT NULL,value TEXT NOT NULL,normalized_number TEXT,unit TEXT,evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),canonical_name TEXT NOT NULL,value_type TEXT NOT NULL CHECK(value_type IN ('number','boolean','string','unknown')),status TEXT NOT NULL CHECK(status IN ('known','unknown')),numeric_value REAL,bool_value INTEGER CHECK(bool_value IN (0,1)),text_value TEXT,observed_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reviews (review_id TEXT PRIMARY KEY,variant_id TEXT NOT NULL REFERENCES variants,source_label TEXT,source_url TEXT,title TEXT,text TEXT NOT NULL,rating REAL,date_text TEXT,duplicate_group_id TEXT NOT NULL,split TEXT,scope TEXT NOT NULL,evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id));
CREATE TABLE IF NOT EXISTS review_duplicate_groups (group_id TEXT PRIMARY KEY,split TEXT,merged_into TEXT REFERENCES review_duplicate_groups(group_id));
CREATE INDEX IF NOT EXISTS evidence_variant_kind ON evidence(variant_id,kind,id);
CREATE INDEX IF NOT EXISTS evidence_observed ON evidence(observed_at);
CREATE INDEX IF NOT EXISTS listings_variant_price ON listings(variant_id,currency,observed_at,price_minor);
CREATE INDEX IF NOT EXISTS offers_variant ON offers(variant_id,offer_id);
CREATE INDEX IF NOT EXISTS observations_offer ON offer_observations(offer_id,context_key,observed_at DESC);
CREATE INDEX IF NOT EXISTS variants_category ON variants(category,detail_status,variant_id);
CREATE INDEX IF NOT EXISTS attributes_filter ON attributes(canonical_name,status,numeric_value,unit,variant_id);
CREATE INDEX IF NOT EXISTS attributes_variant ON attributes(variant_id,canonical_name,observed_at DESC);
CREATE INDEX IF NOT EXISTS reviews_duplicate ON reviews(duplicate_group_id,split);
CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(evidence_id UNINDEXED,variant_id UNINDEXED,title,text,tokenize='unicode61');
CREATE TRIGGER IF NOT EXISTS evidence_insert AFTER INSERT ON evidence BEGIN INSERT INTO evidence_fts(rowid,evidence_id,variant_id,title,text) VALUES(new.id,new.evidence_id,new.variant_id,new.title,new.text); END;
CREATE TRIGGER IF NOT EXISTS evidence_delete AFTER DELETE ON evidence BEGIN DELETE FROM evidence_fts WHERE rowid=old.id; END;
CREATE TRIGGER IF NOT EXISTS evidence_update AFTER UPDATE OF title,text,variant_id ON evidence BEGIN DELETE FROM evidence_fts WHERE rowid=old.id; INSERT INTO evidence_fts(rowid,evidence_id,variant_id,title,text) VALUES(new.id,new.evidence_id,new.variant_id,new.title,new.text); END;
-- REAL is only a non-monetary numeric search index; normalized_number retains exact Decimal text.
-- A cost conflict retains all latest ties. Equal-cost ties deterministically pick one observation ID.
CREATE VIEW IF NOT EXISTS _computed_current_offer_observations AS
WITH latest AS (SELECT offer_id,context_key,MAX(observed_at) AS latest_at FROM offer_observations GROUP BY offer_id,context_key),
 tied AS (SELECT o.* FROM offer_observations o JOIN latest l USING(offer_id,context_key) WHERE o.observed_at=l.latest_at),
 conflicts AS (SELECT offer_id,context_key,COUNT(DISTINCT cost_signature)>1 AS cost_conflict FROM tied GROUP BY offer_id,context_key),
 ranked AS (SELECT t.*,c.cost_conflict,ROW_NUMBER() OVER(PARTITION BY t.offer_id,t.context_key ORDER BY t.observation_id) AS chosen_rank FROM tied t JOIN conflicts c USING(offer_id,context_key))
SELECT observation_id,offer_id,raw_id,evidence_id,observed_at,currency,currency_basis,price_minor,shipping_minor,shipping_status,shipping_scope,tax_minor,tax_status,price_includes_tax,shipping_text,additional_costs,displayed_total_minor,context_json,context_key,cost_signature,condition,store_title,cost_conflict FROM ranked WHERE cost_conflict=1 OR chosen_rank=1;
CREATE VIEW IF NOT EXISTS _computed_current_attributes AS SELECT a.* FROM attributes a WHERE a.observed_at=(SELECT MAX(b.observed_at) FROM attributes b WHERE b.variant_id=a.variant_id AND b.canonical_name=a.canonical_name);

-- Current projections are rebuilt once per import; service SELECTs never scan history.
CREATE TABLE IF NOT EXISTS current_offer_rows AS SELECT * FROM _computed_current_offer_observations WHERE 0;
CREATE INDEX IF NOT EXISTS current_offers_offer ON current_offer_rows(offer_id,context_key);
CREATE INDEX IF NOT EXISTS current_offers_price ON current_offer_rows(currency,cost_conflict,price_minor);
CREATE VIEW IF NOT EXISTS current_offer_observations AS SELECT * FROM current_offer_rows;
CREATE TABLE IF NOT EXISTS current_attribute_rows AS SELECT * FROM _computed_current_attributes WHERE 0;
CREATE INDEX IF NOT EXISTS current_attrs_variant ON current_attribute_rows(variant_id,canonical_name);
CREATE INDEX IF NOT EXISTS current_attrs_filter ON current_attribute_rows(canonical_name,status,numeric_value,unit,variant_id);
CREATE VIEW IF NOT EXISTS current_attributes AS SELECT * FROM current_attribute_rows;
CREATE VIEW IF NOT EXISTS _computed_current_listings AS
WITH latest AS (SELECT variant_id,COALESCE(merchant,'') m,COALESCE(source_url,'') u,MAX(observed_at) t FROM listings GROUP BY variant_id,COALESCE(merchant,''),COALESCE(source_url,'')),
tied AS (SELECT l.* FROM listings l JOIN latest x ON l.variant_id=x.variant_id AND COALESCE(l.merchant,'')=x.m AND COALESCE(l.source_url,'')=x.u AND l.observed_at=x.t)
SELECT t.*,CASE WHEN (SELECT COUNT(DISTINCT COALESCE(currency,'?')||':'||COALESCE(price_minor,'?')) FROM tied x WHERE x.variant_id=t.variant_id AND COALESCE(x.merchant,'')=COALESCE(t.merchant,'') AND COALESCE(x.source_url,'')=COALESCE(t.source_url,''))>1 THEN 1 ELSE 0 END cost_conflict FROM tied t;
CREATE TABLE IF NOT EXISTS current_listing_rows AS SELECT * FROM _computed_current_listings WHERE 0;
CREATE INDEX IF NOT EXISTS current_listings_variant ON current_listing_rows(variant_id,currency,cost_conflict);
CREATE VIEW IF NOT EXISTS current_listings AS SELECT * FROM current_listing_rows;
