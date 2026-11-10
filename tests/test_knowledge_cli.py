import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knowledge.cli import main
from knowledge.credentials import load_api_key


class CLITests(unittest.TestCase):
    def test_plain_credential_is_read_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"credential.txt"
            value="a"*64
            path.write_text(value+"\n")
            before=path.stat().st_mtime_ns
            with patch.dict("os.environ",{"SERPAPI_API_KEY":""}):
                self.assertEqual(load_api_key(path),value)
            self.assertEqual(before,path.stat().st_mtime_ns)

    def test_ambiguous_credential_error_does_not_contain_secret(self):
        value="a"*64+" "+"b"*64
        with patch.dict("os.environ",{"SERPAPI_API_KEY":value}):
            with self.assertRaises(ValueError) as error:
                load_api_key()
        self.assertNotIn("a"*64,str(error.exception))

    def test_verify_failure_returns_nonzero(self):
        output=io.StringIO()
        with patch("knowledge.store.verify_snapshot",return_value={"valid":False,"errors":["changed"]}),contextlib.redirect_stdout(output):
            self.assertEqual(main(["verify","/tmp/fictional-snapshot"]),1)
        self.assertFalse(json.loads(output.getvalue())["valid"])

    def test_plan_is_offline_and_writes_requested_report(self):
        with tempfile.TemporaryDirectory() as folder:
            report=Path(folder)/"plan.json"
            output=io.StringIO()
            with patch("knowledge.acquisition.plan_collection",return_value={"planned_new_details":40}) as planner,contextlib.redirect_stdout(output):
                self.assertEqual(main(["plan","--output",str(report)]),0)
            planner.assert_called_once()
            self.assertEqual(json.loads(report.read_text()),{"planned_new_details":40})

    def test_rtf_conversion_is_captured_not_printed(self):
        import subprocess
        value="c"*64
        result=subprocess.CompletedProcess([],0,stdout=("API key: "+value).encode(),stderr=b"")
        out=io.StringIO()
        with patch.dict("os.environ",{"SERPAPI_API_KEY":""}),patch("knowledge.credentials.subprocess.run",return_value=result) as command,contextlib.redirect_stdout(out):
            self.assertEqual(load_api_key("/tmp/fictional.rtf"),value)
        self.assertEqual(out.getvalue(),"")
        self.assertTrue(command.call_args.kwargs["capture_output"])

    def test_label_command_does_not_replace_frozen_labels_with_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            labels=Path(folder)/"labels.json"
            labels.write_text('{"cases":["frozen"]}')
            with patch("knowledge.evaluation.prepare_development_labels",return_value={"case_count":10}),contextlib.redirect_stdout(io.StringIO()):
                result=main(["label-development","--snapshot-id","snapshot_example","--output",str(labels)])
            self.assertEqual(result,0)
            self.assertEqual(json.loads(labels.read_text()),{"cases":["frozen"]})
