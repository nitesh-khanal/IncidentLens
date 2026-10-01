"""Account-free Python 3.12 setup and launcher for Windows, macOS and Linux."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def environment_python(directory):
    return directory / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def runtime_requirements():
    return dict(line.split('==', 1) for line in (ROOT / 'requirements.txt').read_text().splitlines()
                if line.strip() and not line.startswith('#'))


def dependencies_match(python, expected):
    code = "import importlib.metadata as m,json,sys; p=json.loads(sys.argv[1]); sys.exit(0 if all(m.version(k)==v for k,v in p.items()) else 1)"
    return subprocess.run([str(python), '-c', code, json.dumps(expected)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Set up and verify without starting the browser/server')
    parser.add_argument('--port', type=int, default=8501)
    parser.add_argument('--no-browser', action='store_true', help='Run without automatically opening a browser')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        print('IncidentLens needs Python 3.12 for the shipped models.\nWindows: py -3.12 run.py\nmacOS/Linux: python3.12 run.py', file=sys.stderr)
        return 1
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    # Check the shipped assets before spending time installing packages.
    from scripts.check_setup import verify_files
    try:
        verify_files(ROOT)
        directory = ROOT / '.venv'
        python = environment_python(directory)
        if not python.exists():
            print('Creating a local Python environment…', flush=True)
            venv.EnvBuilder(with_pip=True).create(directory)
        elif subprocess.run([str(python), '-c', 'import sys;sys.exit(sys.version_info[:2]!=(3,12))']).returncode:
            print('The existing .venv uses a different Python version. Rename it and run this launcher again.', file=sys.stderr)
            return 1
        if not dependencies_match(python, runtime_requirements()):
            print('Installing pinned runtime packages. Internet is needed for this setup step only; no account or API key is needed.', flush=True)
            subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements.txt')], cwd=ROOT, check=True)
        env = dict(os.environ)
        env['NLTK_DATA'] = str(ROOT / 'data' / 'nltk')
        env['MPLCONFIGDIR'] = str(ROOT / '.cache' / 'matplotlib')
        env['PYTHONPYCACHEPREFIX'] = str(ROOT / '.cache' / 'python')
        subprocess.run([str(python), '-m', 'scripts.check_setup'], cwd=ROOT, env=env, check=True)
        if args.check:
            return 0
        print(f'Opening IncidentLens at http://127.0.0.1:{args.port}. Press Ctrl+C to stop.', flush=True)
        return subprocess.call([str(python), '-m', 'streamlit', 'run', str(ROOT / 'app.py'),
            '--server.address', '127.0.0.1', '--server.port', str(args.port),
            '--server.headless', 'true' if args.no_browser else 'false',
            '--browser.gatherUsageStats', 'false'], cwd=ROOT, env=env)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f'Setup could not finish: {exc}\nCheck internet during package installation, free disk space, and write access to this project folder. No administrator access is required by the launcher.', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nIncidentLens stopped.')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
