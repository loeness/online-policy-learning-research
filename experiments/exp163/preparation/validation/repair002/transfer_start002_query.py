import base64,datetime,hashlib,io,json,os,pwd,stat,subprocess,zipfile
from pathlib import Path,PurePosixPath
meta=json.loads(base64.b64decode('eyJ0cmFuc2ZlciI6IHsiZGVzdGluYXRpb25fcm9vdCI6ICIvZGF0YS9yb2JvdC9wMDAxL2UxNjMiLCAibWVtYmVyX2NvdW50IjogNywgIm1lbWJlcnMiOiBbeyJieXRlcyI6IDE3OTU1LCAicGF0aCI6ICJvMDAyL2dhdGUucHkiLCAic2hhMjU2IjogImRkZjZkMTIzYWQzMDJjMTM0MTM4NTVkNDM5NWZlMTQ3MTZiNDBjMWQ3MTljOTRmNjkzMTIzYTc3OGE5MWU2NWQifSwgeyJieXRlcyI6IDEyNDgsICJwYXRoIjogIm8wMDIvZ2F0ZV9tYXBwaW5nLmpzb24iLCAic2hhMjU2IjogImJhZmQxMWI0NjlhMjQ2MmM0YmI1MmMyYTRmMmExNmRjMDNmOTQzOGU4YTJjYTU0NjYxMTVmZGRiMGJkZmQyOWEifSwgeyJieXRlcyI6IDExNzEsICJwYXRoIjogIm8wMDIvbGF1bmNoX2NvbW1hbmQuc2giLCAic2hhMjU2IjogIjAzNjYyZTljNTYwODIzOGFhNjRhOWY1NGRjOGQ0ZjdkNTU3ZDM3YWFmNGJjNjQyNzUyOGNhMDIyN2Q2YTBjZGUifSwgeyJieXRlcyI6IDIyNDg4LCAicGF0aCI6ICJvMDAyL3J1bl9zcGVjLmpzb24iLCAic2hhMjU2IjogImQ1NzY0YjgwOTFhYmE5NWI3MjIzZWU3Y2RjYjE4ZDc1MmUxMjNkMTcwNjNiYjBjOTM2Nzc1NzUzZjZlODE4OWQifSwgeyJieXRlcyI6IDE4MjkxLCAicGF0aCI6ICJvMDAyL3J1bm5lci5weSIsICJzaGEyNTYiOiAiOWFlZGYzMTg1M2Y5YWE2ZjI2MzBkNmJhYjllOWRkNzI3NTZhMTM3MjgzMGY4N2Q1YzEzMmIxMmJlN2Y5MTNhZiJ9LCB7ImJ5dGVzIjogMzQ1NzMsICJwYXRoIjogIm8wMDIvc2NpZW50aWZpY19tYW5pZmVzdC5qc29uIiwgInNoYTI1NiI6ICJiODM3MjIyYWZmNTdhYjQ1YWJlZTIyNmU5YWE4YmM5ZDJjZDM0NDQ4MDdmYTM0OGRlZTBjMGMxZWNlNzBkYzU0In0sIHsiYnl0ZXMiOiA1NjMwMSwgInBhdGgiOiAibzAwMi9zdXBlcnZpc29yLnB5IiwgInNoYTI1NiI6ICIwNGM3MTgyYTE5ZjkyNzQxZGY0M2ZjOTVlMWM4Yjg0YzA5MWNkODM5NmJlMjkxNDc1ZTBhNzcwZDMyYzQ2M2MzIn1dLCAic3RhdGUiOiAiTE9DQUxfT05MWV9OT1RfVFJBTlNGRVJSRUQiLCAidW5jb21wcmVzc2VkX2J5dGVzIjogMTUyMDI3fSwgImF1dGhvcml6YXRpb24iOiAiZXdvZ0lDSmpiMjl5WkdsdVlYUnZjbDlrYVhKbFkzUnBkbVVpT2lBaVJURTJNMTlTVlU0d01ESmZNakF5TmpFd01qSmZRVkJRVWs5V1JVUWlMQW9nSUNKbGVHVmpkWFJwYjI1ZllYVjBhRzl5YVhwbFpDSTZJSFJ5ZFdVc0NpQWdJbVY0Y0dWeWFXMWxiblJmYVdRaU9pQWlSVEUyTXlJc0NpQWdJbXhwYldsMGN5STZJSHNLSUNBZ0lDSmhaMmR5WldkaGRHVmZjbVZ0YjNSbFgzQnliMnBsWTNSZllubDBaWE1pT2lBeE1EQXdNREF3TURBd01EQXNDaUFnSUNBaVkyRnRjR0ZwWjI1ZmQyRnNiRjl6WldOdmJtUnpJam9nTVRnd0xBb2dJQ0FnSW1OaGJtUnBaR0YwWlY5bWFYUnpJam9nTXpJc0NpQWdJQ0FpWTNCMVgzTmxZMjl1WkhNaU9pQXhNakFzQ2lBZ0lDQWlaVzUyYVhKdmJtMWxiblJmYVc1MFpYSmhZM1JwYjI1eklqb2dNQ3dLSUNBZ0lDSm5jSFZmWkdWMmFXTmxjeUk2SURBc0NpQWdJQ0FpYm1WM1gyRnlkR2xtWVdOMFgyTmhZMmhsWDNSbGJYQmZZbmwwWlhNaU9pQXlOamcwTXpVME5UWXNDaUFnSUNBaWJuVnRaWEpwWTE5MGFISmxZV1J6WDNCbGNsOTNiM0pyWlhJaU9pQXhMQW9nSUNBZ0luZHZjbXRsY25NaU9pQXhDaUFnZlN3S0lDQWljblZ1WDNOd1pXTmZjMmhoTWpVMklqb2dJbVExTnpZMFlqZ3dPVEZoWW1FNU5XSTNNakl6WldVM1kyUmpZakU0WkRjMU1tVXhNak5rTVRjd05qTmlZakJqT1RNMk56YzFOelV6WmpabE9ERTRPV1FpQ24wSyJ9'))
project=Path('/data/robot/p001');root=project/'e163'
def checked(p):
 p=Path(p);assert p.is_relative_to(root) and p.resolve()==p
 assert not any(q.is_symlink() for q in (p,*p.parents))
 return p
assert pwd.getpwuid(os.geteuid()).pw_name=='robot'
assert not checked(root/'o002').exists() and not checked(root/'run002').exists()
raw=checked(root/'repair002.zip').read_bytes()
assert len(raw)==37912 and hashlib.sha256(raw).hexdigest()=='11a23bab40e945b1d73252b2d75207c99882fc5b83eff2725127dae21044a3c9'
def usage(p):return int(subprocess.check_output(['du','-sb','-x','--',str(p)],text=True,timeout=45).split()[0])
pbytes,ebytes=usage(project),usage(root);v=os.statvfs(root)
assert pbytes+110000000<100000000000 and ebytes+110000000<268435456
assert v.f_bavail*v.f_frsize>110000000 and v.f_favail>1000
expected={r['path']:r for r in meta['transfer']['members']};contents={}
assert len(expected)==7
with zipfile.ZipFile(io.BytesIO(raw)) as z:
 assert len(z.infolist())==7 and {i.filename for i in z.infolist()}==set(expected)
 for item in z.infolist():
  name=item.filename;parts=PurePosixPath(name)
  assert not parts.is_absolute() and '..' not in parts.parts and chr(92) not in name
  assert not item.is_dir() and stat.S_IFMT(item.external_attr>>16)!=stat.S_IFLNK
  assert name.startswith('o002/')
  target=checked(root/name);assert not target.exists()
  data=z.read(item);row=expected[name]
  assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
  contents[name]=data
for name,data in contents.items():
 target=checked(root/name);target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as f:f.write(data)
for name,row in expected.items():
 data=checked(root/name).read_bytes();assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
assert bytes([13]) not in checked(root/'o002/launch_command.sh').read_bytes()
spec_raw=checked(root/'o002/run_spec.json').read_bytes()
assert hashlib.sha256(spec_raw).hexdigest()=='d5764b8091aba95b7223ee7cdcb18d752e123d17063bb0c936775753f6e8189d'
spec=json.loads(spec_raw)
assert pbytes-ebytes==spec['project_bytes_before']
auth_bytes=base64.b64decode(meta['authorization']);auth=json.loads(auth_bytes)
assert auth['execution_authorized'] is True and auth['coordinator_directive']=='E163_RUN002_20261022_APPROVED'
assert auth['run_spec_sha256']==hashlib.sha256(spec_raw).hexdigest() and auth['limits']==spec['limits']
with checked(root/'o002/authorization.json').open('xb') as f:f.write(auth_bytes)
r={'status':'RUN002_TRANSFER_VERIFIED_AUTHORIZATION_CREATED','measured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'members':7,'project_bytes':usage(project),'e163_bytes':usage(root),'authorization_sha256':hashlib.sha256(auth_bytes).hexdigest(),'spec_sha256':hashlib.sha256(spec_raw).hexdigest(),'campaign_started':False,'run002_exists':False}
with checked(root/'o002/admission_start002.json').open('x') as f:json.dump(r,f,indent=2);f.write(chr(10))
b=base64.b64encode(json.dumps(r).encode()).decode()
print('RUN002_TRANSFER_B64_BEGIN')
for i in range(0,len(b),60):print(b[i:i+60])
print('RUN002_TRANSFER_B64_END')
