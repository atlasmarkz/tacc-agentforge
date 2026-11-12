"""Offline fresh-clone, pinned seed installation and TACC job wiring checks."""
import contextlib
import hashlib
import io
import json
import os
import re
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from knowledge.bootstrap import diagnose, install_snapshot
from knowledge.cli import main
from knowledge.service import KnowledgeService
from knowledge.settings import load_settings
from knowledge.store import export_snapshot, ingest_runs

REPO = Path(__file__).resolve().parents[1]


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        environment = {key:value for key,value in os.environ.items() if not key.startswith("KNOWLEDGE_")}
        self.environment = patch.dict(os.environ,environment,clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.temporary = tempfile.TemporaryDirectory(prefix="kb-portable-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.project = self.root / "project"
        (self.project / "configs").mkdir(parents=True)
        run = self.root / "source-run"
        (run / "raw").mkdir(parents=True)
        document = {"product_results": {"title": "Travel backpack 70L",
            "price_insights": {"currency": "USD"},
            "product_attributes": [{"name": "Capacity", "value": "70 L"}],
            "stores": [{"name": "Fixture shop", "title": "Travel backpack 70L", "price": "USD 10.00",
                        "extracted_price": 10,"link": "https://example.com/p1", "shipping": "Free"}]}}
        body = json.dumps(document).encode()
        (run / "raw/one.json").write_bytes(body)
        record = {"call_id": "portable-fixture", "engine": "google_product", "status": "ok",
                  "product_id": "p1", "category": "backpacks", "params": {"gl": "us", "hl": "en"},
                  "requested_at": "2026-01-01T00:00:00Z", "raw_path": "raw/one.json",
                  "sha256": hashlib.sha256(body).hexdigest()}
        (run / "index.jsonl").write_text(json.dumps(record) + "\n")
        queries = self.root / "queries.json"
        queries.write_text('{"queries":[]}')
        db = self.root / "source-data/catalog.sqlite"
        ingest_runs([run],db,queries,REPO / "configs/category_schemas.json")
        snapshot = export_snapshot(db,self.root / "source-snapshots")
        self.snapshot_id = snapshot["snapshot_id"]
        self.archive = self.project / "assets/knowledge/seed.zip"
        self.archive.parent.mkdir(parents=True)
        with zipfile.ZipFile(self.archive,"w",zipfile.ZIP_DEFLATED) as package:
            folder = Path(snapshot["path"])
            for path in folder.rglob("*"):
                if path.is_file():
                    package.write(path,Path(folder.name)/path.relative_to(folder))
        self.config = self.project / "configs/knowledge.json"
        self.value = {"version":1,"data_root":"../team-data","snapshot_id":self.snapshot_id,
                      "seed":{"archive":"assets/knowledge/seed.zip","sha256":hashlib.sha256(self.archive.read_bytes()).hexdigest()},
                      "service":{"cache_size":16,"max_result_bytes":8192,"price_ttl_hours":24}}
        self.save_config()

    def save_config(self):
        self.config.write_text(json.dumps(self.value))

    def settings(self, **kwargs):
        return load_settings(self.config,**kwargs)

    def test_paths_resolve_relative_to_project_not_cwd(self):
        settings=self.settings()
        self.assertEqual(settings.data_root,self.root / "team-data")
        self.assertEqual(settings.seed_archive,self.archive)
        self.assertEqual(settings.service["cache_size"],16)

    def test_explicit_override_beats_environment(self):
        with patch.dict(os.environ,{"KNOWLEDGE_DATA_ROOT":str(self.root/"from-env"),
                                    "KNOWLEDGE_SNAPSHOT_ID":"snapshot_env"}):
            env=load_settings(self.config)
            explicit=load_settings(self.config,data_root=self.root/"explicit",snapshot_id=self.snapshot_id)
        self.assertEqual(env.data_root,self.root/"from-env")
        self.assertEqual(env.snapshot_id,"snapshot_env")
        self.assertEqual(explicit.data_root,self.root/"explicit")
        self.assertEqual(explicit.snapshot_id,self.snapshot_id)

    def test_install_relocated_project_is_complete_and_idempotent(self):
        moved=self.root/"another-location/project"
        shutil.copytree(self.project,moved)
        settings=load_settings(moved/"configs/knowledge.json")
        installed=install_snapshot(settings)
        self.assertTrue(installed["verification"]["valid"])
        service=KnowledgeService(settings.snapshots_root,**settings.service)
        page=service.search_products(settings.snapshot_id,{"category":"backpacks"},limit=1)
        self.assertEqual(page["total_count"],1)
        path=Path(installed["snapshot_path"])/"catalog.sqlite"
        before=path.read_bytes()
        self.assertEqual(install_snapshot(settings)["status"],"already_installed")
        self.assertEqual(path.read_bytes(),before)
        self.assertTrue(diagnose(settings)["ready"])

    def test_checksum_mismatch_leaves_no_snapshot(self):
        self.archive.write_bytes(self.archive.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError,"SHA256"):
            install_snapshot(self.settings())
        self.assertFalse(self.settings().snapshots_root.exists())

    def test_rejects_escape_symlink_duplicate_and_wrong_prefix(self):
        for mode in ("escape","symlink","duplicate","wrong-prefix"):
            with self.subTest(mode=mode):
                archive=self.root/(mode+".zip")
                with zipfile.ZipFile(archive,"w") as package:
                    name=self.snapshot_id+"/file"
                    if mode=="escape":name=self.snapshot_id+"/../../escape"
                    if mode=="wrong-prefix":name="another_snapshot/file"
                    if mode=="symlink":
                        item=zipfile.ZipInfo(name);item.create_system=3;item.external_attr=(stat.S_IFLNK|0o777)<<16
                        package.writestr(item,"../../escape")
                    else:
                        package.writestr(name,"fixture")
                        if mode=="duplicate":
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore",UserWarning);package.writestr(name,"again")
                with self.assertRaises(ValueError):
                    install_snapshot(self.settings(),archive=archive,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        self.assertFalse((self.root/"escape").exists())

    def test_existing_invalid_snapshot_is_not_replaced(self):
        settings=self.settings();install_snapshot(settings)
        db=settings.snapshots_root/settings.snapshot_id/"catalog.sqlite"
        db.write_bytes(b"invalid")
        with self.assertRaisesRegex(ValueError,"refusing to replace"):
            install_snapshot(settings)
        self.assertEqual(db.read_bytes(),b"invalid")

    def test_cli_bootstrap_doctor_and_snapshot_audit(self):
        for args in (["bootstrap"],["doctor"],["audit"]):
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["--settings",str(self.config),*args]),0)
            value=json.loads(output.getvalue())
            if args==["doctor"]:self.assertTrue(value["ready"])
            if args==["audit"]:self.assertEqual(value["counts"]["variants"],1)

    def test_configuration_rejects_path_id_and_invalid_sizes(self):
        for changes in ({"snapshot_id":"../escape"},{"service":{"max_result_bytes":1}},{"version":True}):
            original=json.loads(json.dumps(self.value));self.value.update(changes);self.save_config()
            with self.assertRaises(ValueError):self.settings()
            self.value=original;self.save_config()


class TaccPackagingTests(unittest.TestCase):
    def test_fresh_clone_bootstrap_and_local_tacc_job(self):
        with tempfile.TemporaryDirectory(prefix="kb-clone-") as folder:
            root=Path(folder);clone=root/"repo";clone.mkdir()
            shutil.copytree(REPO/"knowledge",clone/"knowledge",ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
            shutil.copytree(REPO/"configs",clone/"configs")
            shutil.copytree(REPO/"assets/knowledge",clone/"assets/knowledge")
            worker=clone/"deploy/tacc/knowledge-worker"
            shutil.copytree(REPO/"deploy/tacc/knowledge-worker",worker,
                            ignore=shutil.ignore_patterns("_stage","runtime.json","__pycache__"))
            env={key:value for key,value in os.environ.items() if not key.startswith("KNOWLEDGE_")}
            env["KNOWLEDGE_DATA_ROOT"]=str(root/"team-data")
            def command(args,cwd=clone,environment=env):
                result=subprocess.run(args,cwd=cwd,env=environment,text=True,capture_output=True,timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                return result
            # -S excludes site-packages: the clone's core must need only stdlib.
            self.assertEqual(json.loads(command([sys.executable,"-S","-m","knowledge","bootstrap"]).stdout)["status"],"installed")
            self.assertTrue(json.loads(command([sys.executable,"-S","-m","knowledge","doctor"]).stdout)["ready"])
            page=json.loads(command([sys.executable,"-S","-m","knowledge","products","--category","earbuds","--limit","1"]).stdout)
            self.assertTrue(page["candidates"])
            command([sys.executable,str(worker/"stage.py"),"--dry-run"])
            self.assertFalse((worker/"_stage").exists())
            command([sys.executable,str(worker/"stage.py")])
            command(["bash","-n",str(worker/"job.sh")])
            tacc_env={**env,"TACC_WORKDIR":str(worker),"TACC_USERDIR":str(root/"tacc-user"),
                      "SLURM_JOB_ID":"local-smoke","KNOWLEDGE_PYTHON":sys.executable,"KNOWLEDGE_LOCAL_TEST":"1"}
            command(["bash","job.sh"],worker,tacc_env)
            report=root/"tacc-user/tacc-agentforge-results/knowledge-local-smoke/smoke.json"
            result=json.loads(report.read_text())
            self.assertEqual(result["execution_context"],"local_simulation")
            self.assertEqual(result["external_calls"],0)
            self.assertTrue(result["verification"]["valid"])
            self.assertEqual(result["verification"]["evidence_pointers_checked"],3525)
            self.assertTrue(result["copied_to_node_scratch"])
            self.assertTrue(Path(result["persistent_snapshot_path"]).is_dir())
            self.assertFalse(Path(result["query_snapshot_path"]).exists())
            runtime=json.loads((worker/"runtime.json").read_text())
            runtime["mode"]="verify"
            (worker/"runtime.json").write_text(json.dumps(runtime))
            tacc_env["SLURM_JOB_ID"]="local-again"
            command(["bash","job.sh"],worker,tacc_env)
            again=json.loads((report.parents[1]/"knowledge-local-again/verify.json").read_text())
            self.assertEqual(again["bootstrap_status"],"already_installed")
            configuration=(worker/"tuxiv.conf").read_text()
            self.assertIn("cpus-per-task=2",configuration)
            self.assertNotIn("gres=gpu",configuration)
            # The existing harness upload must also contain the new importable package.
            shutil.copytree(REPO/"agent",clone/"agent",ignore=shutil.ignore_patterns(".venv","__pycache__"))
            (clone/"inference").mkdir()
            # The inference modules are exactly those that stage.sh uploads (a hand-kept list here went stale
            # when harness_policy.py was added; PR #12 review), so adding a module there cannot break this again.
            stage_text=(REPO/"deploy/tacc/service/agent-e2e/stage.sh").read_text(encoding="utf-8")
            staged=sorted(set(re.findall(r'"\$root/inference/([A-Za-z_]+\.py)"',stage_text)))
            self.assertIn("harness_policy.py",staged)
            self.assertIn("model_port.py",staged)
            for name in staged:
                shutil.copy2(REPO/"inference"/name,clone/"inference"/name)
            harness=clone/"deploy/tacc/service/agent-e2e"
            harness.mkdir(parents=True)
            shutil.copy2(REPO/"deploy/tacc/service/agent-e2e/stage.sh",harness/"stage.sh")
            command(["bash","stage.sh"],harness)
            self.assertTrue((harness/"pkg/knowledge/service.py").is_file())
            self.assertTrue((harness/"pkg/configs/knowledge.json").is_file())
            for name in staged:
                self.assertTrue((harness/"pkg/inference"/name).is_file(),name)
