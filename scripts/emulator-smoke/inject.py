# Minimal AOSP emulator injectAudio protocol, not sample-accurate scoring.
import wave,time,grpc,sys
import emulator_audio_pb2 as p
import emulator_audio_pb2_grpc as rpc
def audio():
 time.sleep(8)
 with wave.open(sys.argv[1]) as w:
  fmt=p.AudioFormat(samplingRate=w.getframerate(),channels=p.AudioFormat.Mono,format=p.AudioFormat.AUD_FMT_S16,mode=p.AudioFormat.MODE_REAL_TIME)
  for _ in range(5):
   w.rewind()
   while data:=w.readframes(320):
    yield p.AudioPacket(format=fmt,audio=data);time.sleep(.02)
rpc.EmulatorControllerStub(grpc.insecure_channel('localhost:8554')).injectAudio(audio(),timeout=100)
print('injectAudio stream completed',flush=True)
