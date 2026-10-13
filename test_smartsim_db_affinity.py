"""Read-only observation tests: fake Redis, fake /proc, and mocked Slurm only."""
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import smartsim_db_affinity as affinity
import test_smartsim_small_scorep as small


TMP = Path('/hpcwork/ro092286/smartsim/CPP-ML-Interface/tmp/opencode')


class AffinityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='db-affinity-', dir=TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def proc(self, pid, tids=(101, 102)):
        root = self.root / str(pid)
        root.mkdir()
        (root / 'stat').write_text(f'{pid} (redis worker (0)) ' + ' '.join(['S'] + ['0'] * 18 + ['12345']) + '\n')
        for tid in tids:
            task = root / 'task' / str(tid)
            task.mkdir(parents=True)
            (task / 'status').write_text('Name:\tredis-worker\nCpus_allowed:\t0000000f\nCpus_allowed_list:\t0-3\n')
        return root

    def test_four_info_pids_and_all_thread_statuses(self):
        server = socket.socket()
        server.bind(('127.0.0.1', 0))
        server.listen()
        server.settimeout(3)
        self.addCleanup(server.close)
        endpoints = [f'127.0.0.1:{server.getsockname()[1]}'] * 4
        requests = []

        def serve():
            for pid in range(101, 105):
                with server.accept()[0] as connection:
                    requests.append(connection.recv(1024))
                    payload = f'# Server\r\nprocess_id:{pid}\r\n'.encode()
                    connection.sendall(f'${len(payload)}\r\n'.encode() + payload + b'\r\n')

        for pid in range(101, 105):
            self.proc(pid, (pid, pid + 10))
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        records = affinity.snapshot_endpoints(endpoints, self.root)
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertEqual([r['pid'] for r in records], list(range(101, 105)))
        self.assertEqual(requests, [b'*2\r\n$4\r\nINFO\r\n$6\r\nserver\r\n'] * 4)
        for record in records:
            self.assertNotIn('error', record)
            self.assertEqual(record['starttime_ticks'], 12345)
            self.assertEqual(len(record['threads']), 2)
            self.assertTrue(record['hostname'])
            self.assertEqual(record['threads'][0]['cpus_allowed_list'], '0-3')
            self.assertIn('Cpus_allowed_list:', record['threads'][0]['status'])

    def test_errors_are_per_endpoint_and_thread(self):
        root = self.proc(101)
        (root / 'task/102/status').unlink()
        with patch.object(affinity, 'redis_pid', side_effect=[101, OSError('INFO refused')]):
            records = affinity.snapshot_endpoints(['db:1', 'db:2'], self.root)
        self.assertIn('error', records[0]['threads'][1])
        self.assertEqual(records[1]['error'], 'INFO refused')

    def test_local_proc_cpu_masks_readable_without_rebinding(self):
        before = os.sched_getaffinity(0)
        record = affinity.process_snapshot(os.getpid())
        self.assertEqual(os.sched_getaffinity(0), before)
        self.assertTrue(record['threads'])
        self.assertTrue(all(t['cpus_allowed_list'] for t in record['threads']))

    def test_optin_background_gate(self):
        with patch.object(affinity.threading, 'Thread') as thread:
            self.assertIsNone(affinity.start_capture(['db:1'], None, {}))
            self.assertIsNone(affinity.start_capture(['db:1'], None, {'SMARTSIM_DB_AFFINITY_CAPTURE': '0'}))
            thread.assert_not_called()
            affinity.start_capture(['db:1'], '1', {'SMARTSIM_DB_AFFINITY_CAPTURE': '1'})
            self.assertTrue(thread.call_args.kwargs['daemon'])
            thread.return_value.start.assert_called_once()

    def test_remote_host_group_timeout_and_clean_environment(self):
        output = self.root / 'db_affinity.json'
        env = dict(SMARTSIM_DB_AFFINITY_OUTPUT=str(output), SLURM_HET_SIZE='2',
                   SLURM_JOB_ID='42', SCOREP_ENABLE_TRACING='true', SCOREP_ENABLE_PROFILING='true',
                   PATH='/usr/bin:/opt/Score-P/bin', LD_PRELOAD='/opt/libscorep.so /opt/libother.so',
                   LD_AUDIT='/opt/libscorep_audit.so', SMARTSIM_THREADS_PER_QUEUE='2')
        records = [{'endpoint': f'db:{i}', 'hostname': 'dbhost', 'pid': i} for i in range(4)]
        with patch.object(affinity.subprocess, 'run', side_effect=[
                subprocess.CompletedProcess([], 0, 'JobId=43 NodeList=dbhost\n', ''),
                subprocess.CompletedProcess([], 0, 'dbhost\n', ''),
                subprocess.CompletedProcess([], 0, json.dumps(records), '')]) as run:
            affinity.capture([r['endpoint'] for r in records], '1', env)
        self.assertEqual(json.loads(output.read_text())['state'], 'complete')
        calls = run.call_args_list
        self.assertEqual(calls[0].args[0], ['scontrol', 'show', 'job', '42+1', '--oneliner'])
        cmd = calls[-1].args[0]
        for option in ('--overlap', '--exact', '--ntasks=1', '--cpus-per-task=1',
                       '--nodelist=dbhost', '--het-group=1'):
            self.assertIn(option, cmd)
        self.assertFalse(any('bind' in arg for arg in cmd))
        self.assertLessEqual(calls[-1].kwargs['timeout'], 20)
        child = calls[-1].kwargs['env']
        self.assertFalse(any(k.startswith('SCOREP_') for k in child))
        self.assertEqual(child['PATH'], '/usr/bin')
        self.assertEqual(child['LD_PRELOAD'], '/opt/libother.so')
        self.assertEqual(child['LD_AUDIT'], '')
        self.assertEqual(child['SMARTSIM_THREADS_PER_QUEUE'], '2')
        self.assertEqual(env['SCOREP_ENABLE_TRACING'], 'true')

    def test_timeout_and_unavailable_host_record_error_without_raising(self):
        output = self.root / 'db_affinity.json'
        env = dict(SMARTSIM_DB_AFFINITY_OUTPUT=str(output), SLURM_NODELIST='dbhost')
        with patch.object(affinity.subprocess, 'run', side_effect=[
                subprocess.CompletedProcess([], 0, 'dbhost\n', ''),
                subprocess.TimeoutExpired('srun', 20)]), contextlib.redirect_stdout(io.StringIO()) as log:
            affinity.capture(['db:1'], None, env)
        self.assertEqual(json.loads(output.read_text())['state'], 'error')
        self.assertIn('Warning:', log.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            affinity.capture(['db:1'], None, {'SMARTSIM_DB_AFFINITY_OUTPUT': str(output)})
        self.assertIn('nodelist unavailable', json.loads(output.read_text())['error'])

    def test_runner_enables_only_new_small_campaign(self):
        fixture = small.SmartSimSmallTests(methodName='test_small_runtime_matrix_and_metadata')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.small_fixture()
        for campaign in ('smartsim_small_capture_test', 'legacy22'):
            env = fixture.small_env('96p96c4g', 'profile')
            env.update(PILOT_CAMPAIGN_ENV=campaign, SMARTSIM_DB_AFFINITY_CAPTURE='1',
                       SMARTSIM_DB_AFFINITY_OUTPUT='inherited_bad_path')
            result = fixture.run_script('retest_scorep_pilot.zsh', '96p96c4g', 'giantmlp288', 'SMARTSIM', 'pergpu', env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            runtime = next((fixture.mini / f'results/scorep_pilots/{campaign}').glob('*/runtime.env'))
            text = runtime.read_text()
            if campaign.startswith('smartsim_small_'):
                self.assertIn('SMARTSIM_DB_AFFINITY_CAPTURE=1\n', text)
                self.assertIn(f'SMARTSIM_DB_AFFINITY_OUTPUT={runtime.parent}/db_affinity.json\n', text)
            else:
                self.assertNotIn('SMARTSIM_DB_AFFINITY_', text)
        controller = (small.legacy.MINI / 'smartsim_controller.py').read_text()
        self.assertGreater(controller.index('start_capture(address'), controller.index('f.write(",".join(address))'))


if __name__ == '__main__':
    unittest.main()
