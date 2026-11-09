"""Explicit-session log/checkpoint collection; release only after final diagnostics."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import json
import os
import re
import subprocess
import sys
import time
import yaml


@dataclass
class Collector:
    """Copy job logs and checkpoints; verify final archives before releasing its runtime."""
    session: str
    job_id: str
    remote_job_dir: str
    remote_console: str
    remote_diagnostics: str
    output: str
    session_file: str
    executable: str = 'colab'
    auth: str = 'oauth2'
    resume_step: int = 0
    total_steps: int = 982
    retain_steps: list[int] = field(default_factory=lambda: [737, 860, 982])
    snapshot_files: list[str] = field(default_factory=list)
    checkpoint_archives: str | None = None
    interval: float = 90

    def save(self, path: str | Path) -> None:
        """Serialize explicit session and collection settings as YAML."""
        from dataclasses import asdict
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))

    def run(self) -> None:
        if self.interval <= 0 or self.total_steps <= 0 or not self.session or not self.job_id:
            raise ValueError('session, job identity and positive interval/steps required')
        OUT=Path(self.output)
        OUT.mkdir(parents=True, exist_ok=True)
        RUN=vars(self)
        SESSION=self.session
        JOB=self.remote_job_dir
        CLI=[self.executable,'--auth',self.auth,'--config',self.session_file]
        ENV=os.environ.copy()
        def call(*args: str, timeout: int = 180) -> bool:
            """Run a session-specific command and record redacted transport failures."""
            record={'time':time.time(),'operation':args[0]}
            try:
                p=subprocess.run(CLI+list(args)+['--session',SESSION],env=ENV,capture_output=True,timeout=timeout)
                record['returncode']=p.returncode
                if p.returncode:
                    message=(p.stdout+p.stderr).decode(errors='replace')[-4000:]
                    message=re.sub(r'https?://[^\s]+', '[url redacted]', message)
                    message=re.sub(r'eyJ[\w.=-]+', '[token redacted]', message)
                    message=re.sub(r'(?i)(bearer\s+)\S+', r'\1[redacted]', message)
                    record['error']=message
                return p.returncode==0
            except subprocess.TimeoutExpired:
                record['error']=f'TimeoutExpired after {timeout} seconds'
                return False
            finally:
                with (OUT/'transport.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')


        def fetch(remote: str, name: str) -> bool:
            """Replace a local diagnostic file only after a successful download."""
            destination=OUT/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary=destination.with_suffix(destination.suffix+'.pending')
            if call('download',remote,str(temporary)):
                temporary.replace(destination)
                return True
            return False


        def archive(name: str, remote: str) -> bool:
            """Download and verify a remote archive before accepting it locally."""
            destination=OUT/(name+'.tar.gz')
            if destination.exists(): return True
            script=OUT/'archive_remote.py'
            script.write_text(archive_script(remote, '/content/'+name+'.tar.gz',
                                             self.remote_console if remote == self.remote_diagnostics else None))
            prebuilt = self.checkpoint_archives if name.startswith('checkpoint-') else None
            if not prebuilt and not call('exec','--timeout','120','--file',str(script)): return False
            remote_archive = str(Path(prebuilt) / (name+'.tar.gz')) if prebuilt else '/content/'+name+'.tar.gz'
            temporary=destination.with_suffix('.partial')
            if not call('download',remote_archive,str(temporary),timeout=900): return False
            import tarfile
            try:
                with tarfile.open(temporary) as t:
                    for member in t:
                        if member.isfile():
                            f=t.extractfile(member)
                            while f.read(1024*1024): pass
            except Exception:
                return False
            temporary.rename(destination)
            return True

        failures=0
        observations=[]
        baseline=None
        if (OUT/'throughput.jsonl').exists():
            for line in (OUT/'throughput.jsonl').read_text().splitlines():
                row=json.loads(line)
                if row['job_id']==RUN['job_id']:
                    baseline=(row['time'],row['step']); break
        while True:
            # Fetch diagnostics even when the worker status is unavailable or terminal.
            fetch(RUN['remote_console'], 'console.log')
            fetch(RUN['remote_diagnostics']+'/events.jsonl', 'runtime-events.jsonl')
            for name in self.snapshot_files:
                fetch(JOB+'/'+name, 'partial/'+self.job_id+'/'+name)
            temporary=OUT/'status.pending.json'
            if not call('download',JOB+'/status.json',str(temporary)):
                if fetch(RUN['remote_diagnostics']+'/terminal.json', 'terminal.json'):
                    status={'job_id':self.job_id,'state':'failed','error':'worker exited before writing status'}
                else:
                    failures+=1
                    (OUT/'monitor.json').write_text(json.dumps({'state':'connection_retry','failures':failures,'time':time.time()}))
                    if failures >= 5:
                        (OUT/'monitor.json').write_text(json.dumps({'state':'connection_lost','session':SESSION,'failures':failures,'time':time.time(),'recovery':'recurring monitor must reconcile and recover'}))
                    time.sleep(60); continue
            else:
                status=json.loads(temporary.read_text())
                temporary.replace(OUT/'status.json')
            if status['job_id'] != self.job_id:
                raise ValueError('remote job identity differs')
            if fetch(RUN['remote_diagnostics']+'/terminal.json', 'terminal.json') and status['state']=='running':
                status.update(state='failed', error='worker exited without terminal status')
            if status['state'] in ('failed','interrupted'):
                fetch(JOB+'/traceback.txt', 'traceback.txt')
            call('download',JOB+'/log.txt',str(OUT/'training.jsonl'))
            failures=0
            now=time.time()
            step=status.get('step',RUN.get('resume_step',0))
            observations.append((now,step))
            if baseline is None: baseline=(now,step)
            elapsed=now-baseline[0]
            advanced=step-baseline[1]
            rate=advanced/elapsed if elapsed>=300 and advanced>0 else None
            snapshot={'time':now,'session':SESSION,'job_id':status['job_id'],'step':step,'total':self.total_steps,'active_seconds':elapsed,'updates_per_hour':rate*3600 if rate else None,'remaining_seconds':(self.total_steps-step)/rate if rate else None}
            if len(observations)>1:
                previous=observations[-2]
                snapshot['interval_updates_per_hour']=(step-previous[1])*3600/(now-previous[0])
            with (OUT/'throughput.jsonl').open('a') as history: history.write(json.dumps(snapshot)+'\n')
            checkpoint=status.get('checkpoint')
            checkpoint_copies_ready=True
            if checkpoint:
                latest=int(checkpoint.split('-')[-1])
                targets={latest}
                if self.checkpoint_archives:
                    configuration=OUT/'job-config.yaml'
                    if not configuration.exists() and not fetch(JOB+'/config.yaml', 'job-config.yaml'):
                        checkpoint_copies_ready=False
                    settings=(yaml.safe_load(configuration.read_text()) or {}) if configuration.exists() else {}
                    stride=settings.get('arguments', {}).get('checkpoint_every')
                    if isinstance(stride, int) and stride>0:
                        completed=status.get('step', latest)
                        safety=[n for n in range(stride, completed+1, stride) if n>self.resume_step and n not in self.retain_steps]
                        if status['state'] != 'completed':
                            targets.update(safety[-1:])
                        targets.update(n for n in self.retain_steps if self.resume_step<n<=completed)
                for number in sorted(targets):
                    source=JOB+'/'+checkpoint if number==latest else JOB+'/checkpoints/step-'+str(number)
                    if not archive('checkpoint-'+str(number),source): checkpoint_copies_ready=False
                if checkpoint_copies_ready:
                    safety=sorted((p for p in OUT.glob('checkpoint-*.tar.gz') if int(p.name.split('-')[1].split('.')[0]) not in self.retain_steps),key=lambda p:int(p.name.split('-')[1].split('.')[0]))
                    keep=safety[-1:] if status['state']!='completed' else [p for p in safety if p.name==f'checkpoint-{latest}.tar.gz']
                    for old in safety:
                        if old not in keep: old.unlink()
            (OUT/'monitor.json').write_text(json.dumps({'state':'monitoring','session':SESSION,'training_state':status['state'],'step':status.get('step',0),'time':time.time(),'remaining_seconds':snapshot['remaining_seconds'],'updates_per_hour':snapshot['updates_per_hour']}))
            if status['state'] in ('completed','failed','interrupted') and checkpoint_copies_ready:
                if finish(status, fetch, archive, call, JOB, RUN['remote_console'], RUN['remote_diagnostics']):
                    (OUT/'monitor.json').write_text(json.dumps({'state':'archived','training_state':status['state'],'released':True,'time':time.time()}))
                    break
                (OUT/'monitor.json').write_text(json.dumps({'state':'awaiting_final_logs','training_state':status['state'],'time':time.time()}))
            time.sleep(self.interval)


def finish(status: dict[str, Any], fetch: Callable, archive: Callable, call: Callable,
           job: str, console: str, diagnostics: str) -> bool:
    # Worker status can precede Python's final traceback; wait for process exit.
    """Collect final diagnostics and archives before releasing an exited job."""
    if not fetch(diagnostics+'/terminal.json', 'terminal.json'):
        return False
    if not fetch(console, 'console.log') or not fetch(diagnostics+'/events.jsonl', 'runtime-events.jsonl'):
        return False
    if status['state'] in ('failed', 'interrupted'):
        fetch(job+'/traceback.txt', 'traceback.txt')  # May not exist after SIGKILL.
    if not archive('final-diagnostics-'+status['job_id'], diagnostics) or not archive('final-run-'+status['job_id'], job):
        return False
    return call('stop')


def archive_script(source: str, destination: str, console: str | None = None) -> str:
    """Build remote archive code including an optional external console log."""
    code = f'import tarfile\nfrom pathlib import Path\nsource=Path({source!r})\nassert source.exists()\nwith tarfile.open({destination!r}, "w:gz") as t:\n    t.add(source, arcname=source.name)\n'
    if console:
        code += f'    t.add({console!r}, arcname="console.log")\n'
    return code


if __name__ == '__main__':
    Collector(**yaml.safe_load(Path(sys.argv[1]).read_text())).run()
