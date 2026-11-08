import argparse
import json
from pathlib import Path
from .core import audit
from .demo import demo


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description='Small convex logistic sample-weight influence audits')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('demo')
    request = sub.add_parser('audit')
    request.add_argument('input')
    args = parser.parse_args()
    try:
        if args.command == 'demo':
            result = demo()
        else:
            raw = Path(args.input).read_bytes()
            if len(raw) > 100_000:
                raise ValueError('input limit:100000 bytes')
            data = json.loads(raw, object_pairs_hook=strict_object, parse_constant=lambda v: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
            if not isinstance(data, dict) or set(data) != {'version', 'X', 'y', 'query', 'label', 'ridge', 'damping', 'refit'}:
                raise ValueError('expected exact version1 request fields')
            if type(data['version']) is not int or data['version'] != 1:
                raise ValueError('version must be integer1')
            result = audit(data['X'], data['y'], data['query'], data['label'], data['ridge'], data['damping'], data['refit'])
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, TypeError, OverflowError) as exc:
        parser.exit(2, 'error: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
