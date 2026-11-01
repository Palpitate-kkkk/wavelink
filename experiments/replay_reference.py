"""Replay selected v0.21 records; reproduction seeds are NOT new evidence."""
import argparse
import csv
import json
from pathlib import Path
from wavelink.audit import audit_report
from experiments.reference_ber import episode, validate


CASES = [('stationary', 6), ('weak_drift', 18), ('weak_first', 6), ('weaker_first', 18)]


def compare_rows(actual, expected, label):
    if len(actual) != len(expected):
        raise AssertionError(f'{label}: row count {len(actual)} != {len(expected)}')
    for index, (a, b) in enumerate(zip(actual, expected)):
        if set(a) != set(b):
            raise AssertionError(f'{label}: columns differ')
        for key, value in a.items():
            if isinstance(value, float):
                same = abs(value-float(b[key])) <= 1e-12
            else:
                same = str(value) == b[key]
            if not same:
                raise AssertionError(f'{label}: row {index}, {key} differs')


def replay_case(cfg, protocol, scenario, db, seed, records):
    values, costs, events, timing = episode(cfg, protocol, scenario, db, cfg['ebn0_db'].index(db), seed)
    context = dict(scenario=scenario, ebn0_db=db, seed=seed)
    rows = []
    for method, data in values.items():
        for window, (a, b) in {'all': (0, 320), 'late': (224, 320)}.items():
            selected = data[a:b]
            erased = sum(x[1] for x in selected)
            errors = sum(x[0] for x in selected)-erased
            offered = (b-a)*960
            loss = (errors+erased)/offered
            r = dict(context, receiver=method, window=window, offered_bits=offered,
                     accepted_bits=offered-erased, errors=errors, erased_bits=erased, loss=loss)
            for ref, label in [('raw_adaptive', 'raw'), ('raw_ls', 'ls'), ('oracle_adaptive', 'oracle')]:
                r['paired_minus_'+label] = loss-sum(x[0] for x in values[ref][a:b])/offered
            rows.append(r)
    actual = dict(runs=rows, costs=[dict(context, **r) for r in costs],
                  events=[dict(context, **r) for r in events], windows=[dict(context, **timing)])
    counts = {}
    for name, generated in actual.items():
        expected = [r for r in records[name] if all(r[k] == str(v) for k, v in context.items())]
        compare_rows(generated, expected, name)
        counts[name] = len(generated)
    return dict(context, matched_rows=counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('reports/reference'))
    args = parser.parse_args()
    audit = audit_report('.', args.report)
    if not audit['ok']:
        raise SystemExit(json.dumps(audit, ensure_ascii=False))
    cfg = json.loads((args.report/'metadata.json').read_text())['config']
    protocol = validate(cfg)
    records = {name: list(csv.DictReader((args.report/(name+'.csv')).read_text().splitlines())) for name in ('runs', 'costs', 'events', 'windows')}
    results = [replay_case(cfg, protocol, scene, db, cfg['seeds'][0], records) for scene, db in CASES]
    print(json.dumps(dict(ok=True, scope='four saved episodes only; exact integer/hash/event comparison; float tolerance1e-12; no new statistical evidence', cases=results), indent=2))


if __name__ == '__main__':
    main()
