"""Produce the allowlisted byte-extraction query; do not connect or execute it."""
from pathlib import Path
import base64
import json

base = Path(__file__).resolve().parent
transfer = json.loads((base / "transfer_manifest.json").read_text())
freeze = json.loads((base / "execution_freeze001.json").read_text())
metadata = {"transfer": transfer, "archive_sha256": freeze["archive_sha256"],
            "archive_bytes": freeze["archive_bytes"], "run_spec_sha256": freeze["run_spec_sha256"]}
payload = base64.b64encode(json.dumps(metadata).encode()).decode()
query = '''import ast,base64,datetime,hashlib,io,json,os,pwd,stat,subprocess,zipfile
from pathlib import Path,PurePosixPath
meta=json.loads(base64.b64decode('PAYLOAD_PLACEHOLDER'))
project=Path('/data/robot/p001')
root=project/'e163'
def checked(p):
 p=Path(p)
 assert p.is_relative_to(root) and p.resolve()==p
 assert not any(q.is_symlink() for q in (p,*p.parents))
 return p
assert pwd.getpwuid(os.geteuid()).pw_name=='robot'
assert root.resolve()==root and not root.is_symlink()
archive=checked(root/'transfer001.zip')
raw=archive.read_bytes()
assert len(raw)==meta['archive_bytes']
assert hashlib.sha256(raw).hexdigest()==meta['archive_sha256']
manifest=meta['transfer']
assert manifest['destination_root']==str(root)
expected={row['path']:row for row in manifest['members']}
assert len(expected)==manifest['member_count']
def usage(p):
 return int(subprocess.check_output(['du','-sb','-x','--',str(p)],text=True,timeout=45).split()[0])
before_project,before_e=usage(project),usage(root)
v=os.statvfs(root)
peak=manifest['uncompressed_bytes']+32*262144+96*1024*1024
assert before_e+peak<268435456
assert before_project+peak<100000000000
assert v.f_bavail*v.f_frsize>peak and v.f_favail>1000
contents={}
with zipfile.ZipFile(io.BytesIO(raw)) as z:
 assert len(z.infolist())==len(expected)
 assert {i.filename for i in z.infolist()}==set(expected)
 for item in z.infolist():
  name=item.filename
  parts=PurePosixPath(name)
  assert not parts.is_absolute() and '..' not in parts.parts
  assert '\\\\' not in name and not item.is_dir()
  assert stat.S_IFMT(item.external_attr>>16)!=stat.S_IFLNK
  target=checked(root/name)
  assert not target.exists()
  data=z.read(item)
  row=expected[name]
  assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
  contents[name]=data
for name,data in contents.items():
 target=checked(root/name)
 target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as f:f.write(data)
for name,row in expected.items():
 data=checked(root/name).read_bytes()
 assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
for name in ('runner.py','supervisor.py','gate.py'):
 ast.parse(checked(root/'o001'/name).read_text())
spec=checked(root/'o001/run_spec.json').read_bytes()
assert hashlib.sha256(spec).hexdigest()==meta['run_spec_sha256']
s=json.loads(spec)
identity=hashlib.sha256(json.dumps(sorted(s['source_files'],key=lambda r:r['path']),sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert identity==s['source_identity_sha256']
assert not (root/'run001').exists()
assert not (root/'o001/authorization.json').exists()
result={'status':'PREPARATION_TRANSFER_AND_OPAQUE_HASH_SYNTAX_CHECK_PASS','measured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'member_count':len(expected),'source_files':21,'orchestration_code_files':3,'source_identity_sha256':identity,'run_spec_sha256':meta['run_spec_sha256'],'archive_sha256':meta['archive_sha256'],'project_bytes':usage(project),'e163_bytes':usage(root),'filesystem_available_bytes':os.statvfs(root).f_bavail*os.statvfs(root).f_frsize,'authorization_file_exists':False,'run_root_exists':False,'real_fits':0,'prefix_deserializations':0,'metering_prechecks':0,'campaign_started':False,'extraction':'verbatim byte writes; no archive chmod/timestamps'}
receipt=checked(root/'o001/preparation_receipt.json')
with receipt.open('x') as f:json.dump(result,f,indent=2);f.write(chr(10))
data=base64.b64encode(json.dumps(result).encode()).decode()
print('E163_TRANSFER_B64_BEGIN')
for i in range(0,len(data),60):print(data[i:i+60])
print('E163_TRANSFER_B64_END')
'''.replace("PAYLOAD_PLACEHOLDER", payload)
(base / "remote_verify_extract.py").write_text(query, encoding="utf-8")
print(json.dumps({"query_bytes": len(query.encode()), "science_imports": 0, "execution": "NOT_RUN"}))
