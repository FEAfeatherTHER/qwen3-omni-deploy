#!/usr/bin/env python3
"""Resume missing Hugging Face model files; preserve all existing model files."""
import argparse
import json
import os
from pathlib import Path

from check_model import check_model


def download_model(path, repo_id='Qwen/Qwen3-Omni-30B-A3B-Instruct',
                   endpoint='https://huggingface.co', revision='main'):
    root = Path(path).expanduser().resolve()
    try:
        report = check_model(root)
    except ValueError:
        pass
    else:
        return {**report, 'download': 'already complete; no network request'}
    # Weights exceed 10 GB. Settings apply to this process, never the shell/system.
    for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(name, None)
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = '*'
    os.environ['HF_ENDPOINT'] = endpoint.rstrip('/')
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    # Import after process-local settings, since huggingface_hub reads them at import.
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi(endpoint=endpoint).model_info(repo_id, revision=revision, files_metadata=True)
    files = [f for f in info.siblings
             if '/' not in f.rfilename and
             (f.rfilename.endswith(('.json', '.safetensors')) or f.rfilename in ('merges.txt', 'chat_template.jinja'))]
    if not files or not info.sha:
        raise ValueError('Remote repository returned no model files or revision')
    # Validate existing destinations before any downloads. Never replace user files.
    for entry in files:
        target = root / entry.rfilename
        if entry.size is None:
            raise ValueError(f'Remote file size missing: {entry.rfilename}')
        if target.exists() or target.is_symlink():
            if not target.is_file() or target.stat().st_size != entry.size:
                raise ValueError(f'Existing file differs from remote; preserved: {target}. Use a new destination or inspect it manually.')
    root.mkdir(parents=True, exist_ok=True)
    for entry in files:
        target = root / entry.rfilename
        if target.is_file():
            print(f'Reuse {entry.rfilename}', flush=True)
            continue
        print(f'Download {entry.rfilename}', flush=True)
        # The Hub maintains .cache/huggingface/*.incomplete and resumes interrupted HTTP downloads.
        hf_hub_download(repo_id, entry.rfilename, revision=info.sha,
                        endpoint=endpoint, local_dir=root, force_download=False)
    return {**check_model(root), 'revision': info.sha, 'download': 'complete'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_dir')
    parser.add_argument('--repo-id', default='Qwen/Qwen3-Omni-30B-A3B-Instruct')
    parser.add_argument('--endpoint', default='https://huggingface.co', help='For example https://hf-mirror.com')
    parser.add_argument('--revision', default='main', help='Pin a commit when reproducing a deployment')
    args = parser.parse_args()
    try:
        print(json.dumps(download_model(args.model_dir, args.repo_id, args.endpoint, args.revision), indent=2))
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f'Download failed: {exc}\n')


if __name__ == '__main__':
    main()
