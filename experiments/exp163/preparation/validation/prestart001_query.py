import base64,datetime,hashlib,json,os,pwd,subprocess,sys
from pathlib import Path
project=Path('/data/robot/p001');root=project/'e163'
def checked(p):
 p=Path(p)
 assert p.is_absolute() and p.is_relative_to(project)
 assert p.resolve()==p and not any(q.is_symlink() for q in (p,*p.parents))
 return p
assert pwd.getpwuid(os.geteuid()).pw_name=='robot'
assert checked(root).is_dir()
spec_raw=checked(root/'o001/run_spec.json').read_bytes()
assert hashlib.sha256(spec_raw).hexdigest()=='6e7604856c28ce33218f732eb893c1cc56827cab7eb4e51abee21b994f4491e8'
spec=json.loads(spec_raw)
assert not checked(root/'run001').exists()
assert not checked(root/'o001/authorization.json').exists()
def verify(p,row):
 data=checked(p).read_bytes()
 assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],str(p)
for row in spec['source_files']:verify(root/'s001'/row['path'],row)
for row in spec['orchestration_files']:verify(root/'o001'/row['path'],row)
for row in spec['input_files']:verify(row['remote_path'],row)
verify(spec['foundation']['remote_path'],spec['foundation'])
assert hashlib.sha256(checked(root/'o001/scientific_manifest.json').read_bytes()).hexdigest()=='b837222aff57ab45abee226e9aa8bc9d2cd3444807fa348dee0c0c1ece70dc54'
assert hashlib.sha256(checked(root/'transfer001.zip').read_bytes()).hexdigest()=='d4b5b3af9122b15ef70a9fca65a69c57ca84402cd9384518cf37a00e915a093f'
def usage(path):return int(subprocess.check_output(['du','-sb','-x','--',str(path)],text=True,timeout=45).split()[0])
project_bytes,e_bytes=usage(project),usage(root)
v=os.statvfs(root);reserve=109680415
assert project_bytes+reserve<100000000000 and e_bytes+reserve<268435456
assert v.f_bavail*v.f_frsize>reserve and v.f_favail>1000
payload=base64.b64decode('ew0KICAiY29vcmRpbmF0b3JfZGlyZWN0aXZlIjogIkUxNjNfUlVOMDAxXzIwMjYxMDIyX0FQUFJPVkVEIiwNCiAgImV4ZWN1dGlvbl9hdXRob3JpemVkIjogdHJ1ZSwNCiAgImV4cGVyaW1lbnRfaWQiOiAiRTE2MyIsDQogICJsaW1pdHMiOiB7DQogICAgImFnZ3JlZ2F0ZV9yZW1vdGVfcHJvamVjdF9ieXRlcyI6IDEwMDAwMDAwMDAwMCwNCiAgICAiY2FtcGFpZ25fd2FsbF9zZWNvbmRzIjogMTgwLA0KICAgICJjYW5kaWRhdGVfZml0cyI6IDMyLA0KICAgICJjcHVfc2Vjb25kcyI6IDEyMCwNCiAgICAiZW52aXJvbm1lbnRfaW50ZXJhY3Rpb25zIjogMCwNCiAgICAiZ3B1X2RldmljZXMiOiAwLA0KICAgICJuZXdfYXJ0aWZhY3RfY2FjaGVfdGVtcF9ieXRlcyI6IDI2ODQzNTQ1NiwNCiAgICAibnVtZXJpY190aHJlYWRzX3Blcl93b3JrZXIiOiAxLA0KICAgICJ3b3JrZXJzIjogMQ0KICB9LA0KICAicnVuX3NwZWNfc2hhMjU2IjogIjZlNzYwNDg1NmMyOGNlMzMyMThmNzMyZWI4OTNjMWNjNTY4MjdjYWI3ZWI0ZTUxYWJlZTIxYjk5NGY0NDkxZTgiDQp9DQo=')
a=json.loads(payload)
assert a['execution_authorized'] is True and a['limits']==spec['limits']
assert a['run_spec_sha256']==hashlib.sha256(spec_raw).hexdigest()
with checked(root/'o001/authorization.json').open('xb') as f:f.write(payload)
r={'status':'ADMITTED_AUTHORIZATION_CREATED_NOT_STARTED','measured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'account':'robot','project_bytes':project_bytes,'e163_bytes':e_bytes,'available_bytes':v.f_bavail*v.f_frsize,'available_inodes':v.f_favail,'source_files_verified':21,'orchestration_files_verified':4,'input_and_foundation_verified':49,'authorization_sha256':hashlib.sha256(payload).hexdigest(),'run_spec_sha256':a['run_spec_sha256'],'campaign_started':False}
with checked(root/'o001/admission_start001.json').open('x') as f:json.dump(r,f,indent=2);f.write(chr(10))
data=base64.b64encode(json.dumps(r).encode()).decode()
print('E163_START_ADMISSION_B64_BEGIN')
for i in range(0,len(data),60):print(data[i:i+60])
print('E163_START_ADMISSION_B64_END')
