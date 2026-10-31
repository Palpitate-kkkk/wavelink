"""Read-only source/artifact integrity check for reports carrying a manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def audit_report(root, report):
    root = Path(root).resolve()
    report = (root/Path(report)).resolve()
    failures = []
    counts = dict(source_files=0, artifact_files=0)
    try:
        report.relative_to(root)
        meta = json.loads((report/'metadata.json').read_text())
    except (ValueError, OSError, UnicodeError) as exc:
        return dict(ok=False, **counts, failures=['cannot read in-root metadata: '+str(exc)])
    if not isinstance(meta, dict):
        return dict(ok=False, **counts, failures=['metadata must be a JSON object'])
    for field, base, counter in [('source_sha256', root, 'source_files'), ('artifact_sha256', report, 'artifact_files')]:
        manifest = meta.get(field)
        if not isinstance(manifest, dict) or not manifest:
            failures.append('missing/nonempty manifest required: '+field)
            continue
        for name, expected in manifest.items():
            if not isinstance(name, str) or not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
                failures.append('invalid manifest entry: '+str(name))
                continue
            path = (base/name).resolve()
            try:
                path.relative_to(base)
                if Path(name).is_absolute():
                    raise ValueError('absolute path')
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                counts[counter] += 1
                if actual != expected:
                    failures.append('hash mismatch: '+field+'/'+name)
            except (ValueError, OSError) as exc:
                failures.append('unreadable/escaping path: '+name+' ('+str(exc)+')')
    return dict(ok=not failures, **counts, failures=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--report', type=Path, default=Path('reports/reference'))
    args = parser.parse_args()
    result = audit_report(args.root, args.report)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['ok'] else 1)


if __name__ == '__main__':
    main()
