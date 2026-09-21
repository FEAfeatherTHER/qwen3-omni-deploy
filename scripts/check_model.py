#!/usr/bin/env python3
"""Read model metadata and safetensors headers without loading tensors or hashing."""
import argparse
import json
import math
from pathlib import Path
import struct


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f'Invalid or missing {path.name}: {exc}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'{path.name} must contain a JSON object')
    return value


def check_model(path):
    root = Path(path).expanduser().resolve()
    for name in ('config.json', 'tokenizer_config.json', 'preprocessor_config.json'):
        read_json(root / name)
    tokenizer_config = read_json(root / 'tokenizer_config.json')
    template = tokenizer_config.get('chat_template')
    if not template and (root / 'chat_template.json').is_file():
        template = read_json(root / 'chat_template.json').get('chat_template')
    if not template and (root / 'chat_template.jinja').is_file():
        try:
            template = (root / 'chat_template.jinja').read_text(encoding='utf-8').strip()
        except (OSError, UnicodeError) as exc:
            raise ValueError(f'Invalid chat_template.jinja: {exc}') from exc
    if not isinstance(template, (str, list)) or not template:
        raise ValueError('Missing or empty chat_template for chat inference')
    if (root / 'tokenizer.json').is_file():
        read_json(root / 'tokenizer.json')
    else:
        read_json(root / 'vocab.json')
        merges = root / 'merges.txt'
        if not merges.is_file() or merges.stat().st_size == 0:
            raise ValueError('Missing or empty tokenizer merges.txt')
    index = read_json(root / 'model.safetensors.index.json')
    mapping = index.get('weight_map')
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('Model index has no weight_map')
    shards = {}
    for tensor, filename in mapping.items():
        if not isinstance(tensor, str) or not isinstance(filename, str):
            raise ValueError('Invalid tensor name or shard path in weight_map')
        candidate = root / filename
        if Path(filename).is_absolute() or '..' in Path(filename).parts or candidate.suffix != '.safetensors':
            raise ValueError(f'Invalid shard path: {filename}')
        # Symlinks are common in the Hugging Face cache and are intentionally accepted.
        shards.setdefault(filename, set()).add(tensor)
    total = 0
    file_bytes = 0
    widths = {'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
              'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2, 'I32': 4, 'U32': 4,
              'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8}
    for filename, expected in sorted(shards.items()):
        shard = root / filename
        try:
            size = shard.stat().st_size
            with shard.open('rb') as f:
                prefix = f.read(8)
                if len(prefix) != 8:
                    raise ValueError(f'Truncated safetensors header: {filename}')
                length = struct.unpack('<Q', prefix)[0]
                if not 2 <= length <= min(100_000_000, size - 8):
                    raise ValueError(f'Invalid header size: {filename}')
                header = json.loads(f.read(length))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f'Invalid or missing shard {filename}: {exc}') from exc
        if not isinstance(header, dict):
            raise ValueError(f'Invalid header: {filename}')
        actual = set(header) - {'__metadata__'}
        if actual != expected:
            raise ValueError(f'{filename}: index/header tensors differ; missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}')
        spans = []
        for name in actual:
            info = header[name]
            if not isinstance(info, dict):
                raise ValueError(f'Invalid tensor metadata: {name}')
            offsets, shape, dtype = info.get('data_offsets'), info.get('shape'), info.get('dtype')
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(x) is not int for x in offsets)
                    or not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape)
                    or not isinstance(dtype, str) or dtype not in widths):
                raise ValueError(f'Invalid tensor shape/dtype/offset: {name}')
            start, end = offsets
            if start < 0 or end < start or end - start != math.prod(shape) * widths[dtype]:
                raise ValueError(f'Invalid tensor size/offset: {name}')
            spans.append((start, end))
        cursor = 0
        for start, end in sorted(spans):
            if start != cursor:
                raise ValueError(f'Overlapping or missing tensor data offsets: {filename}')
            cursor = end
        if cursor != size - 8 - length:
            raise ValueError(f'Truncated or excess shard size: {filename}')
        total += cursor
        file_bytes += size
    metadata = index.get('metadata', {})
    if not isinstance(metadata, dict):
        raise ValueError('Invalid index metadata')
    if 'total_size' in metadata and metadata['total_size'] != total:
        raise ValueError(f'Index total_size {metadata["total_size"]} differs from tensor bytes {total}')
    return {'model_dir': str(root), 'shards': len(shards), 'tensors': len(mapping),
            'tensor_bytes': total, 'file_bytes': file_bytes, 'status': 'ok'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir')
    args = parser.parse_args()
    try:
        print(json.dumps(check_model(args.model_dir), indent=2, ensure_ascii=False))
    except ValueError as exc:
        parser.exit(1, f'Model check failed: {exc}\n')


if __name__ == '__main__':
    main()
