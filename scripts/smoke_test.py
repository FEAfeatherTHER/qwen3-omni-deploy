#!/usr/bin/env python3
"""Exercise a running service with real text/audio and concurrent requests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from client import chat, extract_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8001/v1')
    parser.add_argument('--audio', required=True, help='A real short speech recording')
    parser.add_argument('--second-audio', required=True, help='A short sound recording')
    parser.add_argument('--long-audio', required=True, help='A recording of approximately 60 seconds')
    parser.add_argument('--report', required=True, help='New JSON report path')
    args = parser.parse_args()
    target = Path(args.report)
    if target.exists():
        parser.error('Report already exists; choose a new path')
    for path in (args.audio, args.second_audio, args.long_audio):
        if not Path(path).is_file():
            parser.error(f'Audio file does not exist: {path}')
    results = []

    def request(name, prompt, paths=()):
        started = time.monotonic()
        response = chat(prompt, paths, base_url=args.base_url, max_tokens=256)
        text = extract_text(response)
        if not text.strip():
            raise AssertionError(f'{name}: empty answer')
        if response['choices'][0].get('finish_reason') != 'stop':
            raise AssertionError(f'{name}: incomplete answer: {response["choices"][0].get("finish_reason")}')
        if response['choices'][0]['message'].get('audio'):
            raise AssertionError(f'{name}: unexpected audio output')
        return {'case': name, 'seconds': round(time.monotonic() - started, 3),
                'text': text, 'response': response}

    started = datetime.now(timezone.utc).isoformat()
    error = None
    try:
        cases = [
            ('text', '请只回答：服务正常。', []),
            ('transcription', '请将这段中文语音转换为纯文本。', [args.audio]),
            ('sound_description', '请用一句中文描述听到的声音。', [args.second_audio]),
            ('multi_audio', '依次简短描述这四段音频的内容，每段一句话。',
             [args.audio, args.second_audio, args.audio, args.second_audio]),
            ('audio_60s', '请用两句话概括这段音频的内容，不必逐字转写。', [args.long_audio]),
        ]
        for name, prompt, paths in cases:
            results.append(request(name, prompt, paths))
            print(f'{name}: passed ({results[-1]["seconds"]}s)', flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(request, f'concurrent_{i+1}', '请用一句话描述这段音频。',
                                   [args.audio if i % 2 == 0 else args.second_audio]) for i in range(4)]
            for future in futures:
                results.append(future.result())
        print('4 concurrent audio requests: passed', flush=True)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(error, file=sys.stderr)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as handle:
        json.dump({'started_at': started, 'base_url': args.base_url, 'passed': error is None,
                   'error': error, 'results': results}, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return 1 if error else 0


if __name__ == '__main__':
    sys.exit(main())
