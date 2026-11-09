"""Tiny staged pipeline: generate on A100, train on L4, compare on A100."""

from pathlib import Path
import json
import subprocess
import sys

from tc.jobs import Compare, Evaluate, Generate, Job, Prepare, Train
from utils.eval.benchmarks import NUMERIC_INSTRUCTION


def main(root: Path, stage: str) -> None:
    import torch

    expected = {'generate': 'A100', 'train': 'L4', 'compare': 'A100'}[stage]
    if torch.cuda.device_count() != 1 or expected not in torch.cuda.get_device_name(0):
        raise ValueError(f'{stage} requires one {expected}')
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / 'ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}

    def run(name: str, job: Job) -> Path:
        job.name, job.jobs_dir = name, str(root / 'jobs')
        config = root / f'{name}.yaml'
        job.save(config)
        log = root / f'{name}.log'
        with log.open('w') as stream:
            result = subprocess.run([sys.executable, '-m', 'tc', str(config)],
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {log}')
        directory = Path(log.read_text().splitlines()[0])
        ledger[name] = str(directory)
        (root / 'ledger.json').write_text(json.dumps(ledger, indent=2))
        print(name, directory, flush=True)
        return directory

    if stage == 'generate':
        questions = [dict(id=f'smoke:{i}', dataset='svamp', grader='numeric', split='train',
                          seed_index=i, answer=str(answer),
                          messages=[{'role': 'user', 'content': question + NUMERIC_INSTRUCTION}])
                     for i, (question, answer) in enumerate([('What is 2 + 2?', 4), ('What is 7 times 3?', 21)])]
        source = root / 'questions.jsonl'
        source.write_text(''.join(json.dumps(q) + '\n' for q in questions))
        prepared = run('prepare', Prepare(input=[str(source)]))
        inputs = [str(prepared / 'outputs/questions.jsonl')]
        baseline = run('generate-base', Generate(input=inputs, responses=4))
        graded = run('evaluate-base', Evaluate(input=[str(baseline)]))
        return

    inputs = [str(Path(ledger['prepare']) / 'outputs/questions.jsonl')]
    graded = Path(ledger['evaluate-base'])
    for algorithm in ('simpo', 'dpo'):
        if stage == 'train':
            trained = run(f'train-{algorithm}', Train(groups={'input': str(graded)},
                          algorithm=algorithm, steps=2, batch_size=2, gradient_accumulation=2,
                          checkpoint_every=1, retain_step=[1, 2]))
            continue
        trained = Path(ledger[f'train-{algorithm}'])
        candidate = run(f'generate-{algorithm}', Generate(input=inputs, responses=4,
                        adapter=str(trained / 'outputs/adapter')))
        evaluated = run(f'evaluate-{algorithm}', Evaluate(input=[str(candidate)]))
        run(f'compare-{algorithm}', Compare(input=[str(graded), str(evaluated)]))
    if stage == 'train':
        return
    (root / 'completed.json').write_text(json.dumps({'passed': True, 'jobs': ledger}, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(), sys.argv[2])
