"""Collect exact upstream tests against installed code, never source imports."""
import sys,json,shutil,hashlib,platform
from pathlib import Path
import skimage,numpy,scipy
source=Path(sys.argv[1]).resolve()
installed=Path(skimage.__file__).resolve().parent
assert 'site-packages' in str(installed), str(installed)
manifest=[]
if (source/'tests/skimage').is_dir():
    target=source/'tests'
else:
    assert (source/'skimage').is_dir()
    for tests in (source/'skimage').rglob('tests'):
        if not tests.is_dir(): continue
        dest=installed/tests.relative_to(source/'skimage')
        shutil.copytree(tests,dest,dirs_exist_ok=True)
        for f in tests.rglob('*'):
            if f.is_file():
                copied=dest/f.relative_to(tests)
                assert hashlib.sha256(f.read_bytes()).digest()==hashlib.sha256(copied.read_bytes()).digest()
                manifest.append({'source':str(f.relative_to(source)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    assert manifest
    target=installed
result={'imported':str(installed),'version':skimage.__version__,'numpy':numpy.__version__,'scipy':scipy.__version__,'machine':platform.machine(),'test_root':str(target),'staged_exact_tests':manifest}
Path(sys.argv[2]).write_text(json.dumps(result,indent=2))
print(target)
