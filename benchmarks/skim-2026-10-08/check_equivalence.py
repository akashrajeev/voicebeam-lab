exec(open('/home/sandbox/voicefix/skim-host/export_skim.py').read().split('with torch.no_grad():')[0])
with torch.no_grad():
 x=torch.randn(1,2004)*.03;feat,_=enc(x,torch.tensor([x.shape[1]]));off,_,_=sep(feat,torch.tensor([feat.shape[1]]));st=None;buf=[[],[]]
 for f in feat.split(1,dim=1):
  yy,st,_=sep.forward_streaming(f,st)
  for j in range(2):buf[j].append(yy[j])
 for j in range(2):
  stream=torch.cat(buf[j],dim=1);delta=(off[j]-stream).abs().max(dim=2).values[0];print('latent',j,'max',delta.max().item(),'first',[(i,delta[i].item()) for i in [0,1,49,50,51,99,100,101,150,200,300,400]])
 yyoff=dec(off[0],torch.tensor([x.shape[1]]))[0]
 yystream=dec(torch.cat(buf[0],dim=1),torch.tensor([x.shape[1]]))[0];print('wave',float((yyoff-yystream).abs().max()))
