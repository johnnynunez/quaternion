"""Strict native ARM64 wheel audit including mutation negatives."""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import zipfile
from email.parser import Parser


def check_pe(data):
    if len(data)<64 or data[:2]!=b'MZ': raise ValueError('invalid PE DOS header')
    off=struct.unpack_from('<I',data,0x3c)[0]
    if off+6>len(data) or data[off:off+4]!=b'PE\0\0': raise ValueError('invalid PE signature')
    machine=struct.unpack_from('<H',data,off+4)[0]
    if machine!=0xaa64: raise ValueError(f'PE Machine {machine:#x} is not native ARM64')
    return machine


def audit(path, version):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)): raise ValueError('duplicate ZIP path')
        records=[n for n in names if n.endswith('.dist-info/RECORD')]
        if len(records)!=1: raise ValueError('RECORD count')
        rows=list(csv.reader(io.StringIO(z.read(records[0]).decode())))
        if set(r[0] for r in rows)!=set(names): raise ValueError('RECORD coverage')
        for name,digest,size in rows:
            if name==records[0]:
                if digest or size: raise ValueError('RECORD self hash')
                continue
            data=z.read(name)
            if not digest or not size: raise ValueError('RECORD missing digest')
            alg,value=digest.split('=',1)
            calc=base64.urlsafe_b64encode(hashlib.new(alg,data).digest()).decode().rstrip('=')
            if calc!=value or len(data)!=int(size): raise ValueError('RECORD mismatch '+name)
        metadata=Parser().parsestr(z.read(next(n for n in names if n.endswith('.dist-info/METADATA'))).decode())
        if metadata['Version']!=version: raise ValueError('version mismatch '+metadata['Version'])
        pe=[]
        for name in names:
            if name.lower().endswith(('.pyd','.dll','.exe')):
                data=z.read(name)
                pe.append({'path':name,'machine':hex(check_pe(data)),'sha256':hashlib.sha256(data).hexdigest()})
        if not pe: raise ValueError('no native payload')
    return {'wheel':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'version':version,'record_entries':len(rows),'pe':pe}


def main():
    wheel=Path(sys.argv[1]); version=sys.argv[2]; out=Path(sys.argv[3])
    result=audit(wheel,version)
    with zipfile.ZipFile(wheel) as z: payload={n:z.read(n) for n in z.namelist()}
    target=result['pe'][0]['path']
    corrupt=wheel.parent/'negative-record.whl'
    with zipfile.ZipFile(corrupt,'w') as z:
        for n,v in payload.items(): z.writestr(n,v+b'corrupt' if n==target else v)
    try:
        audit(corrupt,version)
    except ValueError as e:
        assert 'RECORD mismatch' in str(e); result['record_negative']=str(e)
    else: raise AssertionError('corrupted RECORD accepted')
    corrupt.unlink()
    bad=bytearray(payload[target]); off=struct.unpack_from('<I',bad,0x3c)[0]; struct.pack_into('<H',bad,off+4,0x8664)
    try: check_pe(bytes(bad))
    except ValueError as e: result['pe_negative']=str(e)
    else: raise AssertionError('AMD64 payload accepted')
    out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
