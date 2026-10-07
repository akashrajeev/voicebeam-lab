#!/usr/bin/env python3
"""Match 4j JNI ELF symbol version to sherpa's ORT core. C API20 retained.
Requires same-size version names. Updates GNU verneed auxiliary ELF hash too.
"""
import struct,sys,zipfile,io
p=sys.argv[1];old=b'VERS_1.20.0';new=b'VERS_1.28.2'
def elfhash(s):
 h=0
 for v in s:
  h=(h<<4)+v;g=h&0xf0000000
  if g:h^=g>>24
  h&=~g
 return h&0xffffffff
with zipfile.ZipFile(p) as src:
 rows=[(i,src.read(i)) for i in src.infolist()]
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as dst:
 for info,data in rows:
  if info.filename.endswith('/libonnxruntime4j_jni.so'):
   b=bytearray(data);assert b[:4]==b'\x7fELF'
   # Version auxiliary hash values are unique for this dependency; explicit old-string count gate.
   assert old in b
   oh=struct.pack('<I',elfhash(old));nh=struct.pack('<I',elfhash(new))
   assert oh in b
   b=b.replace(old,new).replace(oh,nh)
   data=bytes(b)
  dst.writestr(info,data)
