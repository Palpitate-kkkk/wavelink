import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from wavelink.audit import audit_report


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.report = self.root/'reports'/'test'
        self.report.mkdir(parents=True)
        (self.root/'source.py').write_text('source')
        (self.report/'runs.csv').write_text('results')
        h = lambda s: hashlib.sha256(s.encode()).hexdigest()
        self.meta = dict(source_sha256={'source.py': h('source')}, artifact_sha256={'runs.csv': h('results')})
        self.write()

    def write(self):
        (self.report/'metadata.json').write_text(json.dumps(self.meta))

    def test_consistent_report_and_read_only(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        result = audit_report(self.root, 'reports/test')
        self.assertEqual(result, dict(ok=True, source_files=1, artifact_files=1, failures=[]))
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_source_and_artifact_tampering_detected(self):
        for path in (self.root/'source.py', self.report/'runs.csv'):
            original = path.read_bytes()
            path.write_text('changed')
            self.assertFalse(audit_report(self.root, 'reports/test')['ok'])
            path.write_bytes(original)

    def test_escaping_path_and_symlink_rejected(self):
        (self.report/'link').symlink_to(self.root/'source.py')
        for name in ('../../source.py', 'link', str(self.root/'source.py')):
            self.meta['artifact_sha256'] = {name: hashlib.sha256(b'source').hexdigest()}
            self.write()
            self.assertFalse(audit_report(self.root, 'reports/test')['ok'])

    def test_missing_invalid_and_nonobject_metadata_fail_cleanly(self):
        for meta in ([], None, {}, {'source_sha256': {'source.py':'bad'}}):
            self.meta = meta
            self.write()
            self.assertFalse(audit_report(self.root, 'reports/test')['ok'])
        (self.report/'metadata.json').write_text('{')
        self.assertFalse(audit_report(self.root, 'reports/test')['ok'])
        self.assertFalse(audit_report(self.root, '../outside')['ok'])


if __name__ == '__main__':
    unittest.main()
