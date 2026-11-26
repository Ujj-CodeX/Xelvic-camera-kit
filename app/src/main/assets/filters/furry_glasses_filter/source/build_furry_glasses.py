#!/usr/bin/env python3
"""Rebuild Furry Glasses GLB LODs from the package's procedural authoring parameters.
Run from the extracted package root: python source/build_furry_glasses.py
Requires: numpy, Pillow, trimesh.
After geometry changes, refresh SHA-256/byte records in manifest.json and validation/report.json.
"""
from pathlib import Path
import math, json, struct
import numpy as np
from PIL import Image
import trimesh

ROOT=Path(__file__).resolve().parents[1]
TEXTURE=Image.open(ROOT/'assets/furry_plush_pink_srgb.png').convert('RGB')
EYES=[(-0.232,0.0),(0.232,0.0)]
OUTER=(0.222,0.185); INNER=(0.142,0.106); EXP=4.4
LODS={'a':128,'b':88,'c':60}

def contour(cx,cy,hw,hh,n,count):
    t=np.linspace(0,2*np.pi,count,endpoint=False); c=np.cos(t); s=np.sin(t)
    return np.stack([cx+hw*np.sign(c)*np.abs(c)**(2/n),cy+hh*np.sign(s)*np.abs(s)**(2/n)],axis=1)
def uv(x,y,z=0): return [float(np.clip((x+0.56)/1.12,.001,.999)),float(np.clip((y+0.27)/.54,.001,.999))]
def ring(v,f,u,cx,cy,N):
    out=contour(cx,cy,*OUTER,EXP,N); inn=contour(cx,cy,*INNER,EXP,N)
    front=[]
    for a,z in zip([0,.12,.34,.66,.88,1],[.018,.068,.09,.09,.066,.018]):
        ids=[]
        for x,y in out*(1-a)+inn*a: ids.append(len(v));v.append([x,y,z]);u.append(uv(x,y))
        front.append(ids)
    for k in range(len(front)-1):
        A,B=front[k:k+2]
        for i in range(N):
            j=(i+1)%N;f.extend([[A[i],A[j],B[j]],[A[i],B[j],B[i]]])
    back=[]
    for a,z in zip([0,.18,.5,.82,1],[-.035,-.055,-.06,-.05,-.025]):
        ids=[]
        for x,y in out*(1-a)+inn*a: ids.append(len(v));v.append([x,y,z]);u.append(uv(x,y))
        back.append(ids)
    for k in range(len(back)-1):
        A,B=back[k:k+2]
        for i in range(N):
            j=(i+1)%N;f.extend([[A[i],B[j],A[j]],[A[i],B[i],B[j]]])
    for A,B in [(front[0],back[0]),(front[-1],back[-1])]:
        for i in range(N):
            j=(i+1)%N
            if A is front[0]: f.extend([[A[i],B[j],A[j]],[A[i],B[i],B[j]]])
            else: f.extend([[A[i],A[j],B[j]],[A[i],B[j],B[i]]])
def tube(v,f,u,path,radii,N=16):
    path=np.asarray(path,dtype=float); rings=[]
    for i,p in enumerate(path):
        tangent=(path[1]-path[0]) if i==0 else ((path[-1]-path[-2]) if i==len(path)-1 else path[i+1]-path[i-1])
        tangent/=np.linalg.norm(tangent)+1e-9; ref=np.array([0.,1.,0.])
        if abs(np.dot(tangent,ref))>.92:ref=np.array([1.,0.,0.])
        b1=np.cross(tangent,ref);b1/=np.linalg.norm(b1)+1e-9;b2=np.cross(tangent,b1);b2/=np.linalg.norm(b2)+1e-9
        r=radii[i] if isinstance(radii,(list,tuple,np.ndarray)) else radii; ids=[]
        for k in range(N):
            a=2*np.pi*k/N;q=p+r*(np.cos(a)*b1+np.sin(a)*b2);ids.append(len(v));v.append(q.tolist());u.append(uv(q[0],q[1]))
        rings.append(ids)
    for A,B in zip(rings,rings[1:]):
        for k in range(N):
            j=(k+1)%N;f.extend([[A[k],B[k],B[j]],[A[k],B[j],A[j]]])
    for ids,p,rev in [(rings[0],path[0],True),(rings[-1],path[-1],False)]:
        center=len(v);v.append(p.tolist());u.append(uv(*p))
        for k in range(N):
            j=(k+1)%N;f.append([center,ids[j],ids[k]] if rev else [center,ids[k],ids[j]])
def force_u16(blob):
    magic,ver,total=struct.unpack_from('<4sII',blob,0);assert magic==b'glTF' and ver==2 and total==len(blob)
    jl,jt=struct.unpack_from('<II',blob,12);doc=json.loads(blob[20:20+jl].decode().rstrip(' \t\r\n\0'));bh=20+jl
    bl,bt=struct.unpack_from('<II',blob,bh);assert bt==0x004E4942;binary=bytearray(blob[bh+8:bh+8+bl]);changed=False
    for me in doc.get('meshes',[]):
      for pr in me['primitives']:
        acc=doc['accessors'][pr['indices']]
        if acc['componentType']==5123:continue
        vi=acc['bufferView'];view=doc['bufferViews'][vi];off=view.get('byteOffset',0);old=view['byteLength'];n=acc['count']
        raw=np.frombuffer(binary[off:off+old],dtype='<u4',count=n);assert int(raw.max())<=65535
        packed=raw.astype('<u2').tobytes();delta=old-len(packed);binary=bytearray(binary[:off]+packed+binary[off+old:]);view['byteLength']=len(packed);acc['componentType']=5123
        for ix,bv in enumerate(doc['bufferViews']):
            if ix!=vi and bv.get('byteOffset',0)>=off+old:bv['byteOffset']-=delta
        changed=True
    if not changed:return blob
    doc['buffers'][0]['byteLength']=len(binary);jb=json.dumps(doc,separators=(',',':')).encode();jb+=b' '*((4-len(jb)%4)%4);binary+=b'\0'*((4-len(binary)%4)%4)
    size=12+8+len(jb)+8+len(binary)
    return struct.pack('<4sII',b'glTF',2,size)+struct.pack('<II',len(jb),0x4E4F534A)+jb+struct.pack('<II',len(binary),0x004E4942)+binary

def build(tier,N):
    v=[];f=[];uvs=[]
    for cx,cy in EYES:ring(v,f,uvs,cx,cy,N)
    # Narrow bridge ends at the inner rims; no tube volume enters either eye aperture.
    bridge_v=[];bridge_f=[];bridge_uv=[]
    tube(bridge_v,bridge_f,bridge_uv,[(-.09,-.004,.005),(-.058,-.020,.030),(0.,-.032,.038),(.058,-.020,.030),(.09,-.004,.005)],[.038,.042,.043,.042,.038],20)
    bridge=np.asarray(bridge_v);bridge[:,0]*=.69;off=len(v);v.extend(bridge.tolist());uvs.extend(bridge_uv);f.extend((np.asarray(bridge_f)+off).tolist())
    for sign in (-1,1):tube(v,f,uvs,[(sign*.075,-.105,-.065),(sign*.082,-.125,-.085),(sign*.092,-.137,-.095)],[.027,.032,.025],16)
    for sign in (-1,1):tube(v,f,uvs,[(sign*.425,-.010,-.008),(sign*.465,-.012,-.045),(sign*.505,-.028,-.115),(sign*.535,-.055,-.220),(sign*.535,-.086,-.335),(sign*.505,-.105,-.390)],[.036,.033,.029,.027,.026,.024],max(10,N//8))
    mesh=trimesh.Trimesh(vertices=np.asarray(v,dtype=np.float32),faces=np.asarray(f,dtype=np.uint32),process=True)
    mesh.visual=trimesh.visual.texture.TextureVisuals(uv=np.asarray(uvs,dtype=np.float32),image=TEXTURE)
    mesh.visual.material=trimesh.visual.material.PBRMaterial(name='FurryGlasses_PinkPlush_SRGB',baseColorFactor=[255,255,255,255],baseColorTexture=TEXTURE,metallicFactor=0,roughnessFactor=.92,alphaMode='OPAQUE',doubleSided=True)
    scene=trimesh.Scene();scene.add_geometry(mesh,geom_name='Furry_Glasses_OneCombinedMesh',node_name='FurryGlasses_EyeCenter_Pivot')
    blob=force_u16(trimesh.exchange.gltf.export_glb(scene,include_normals=True))
    out=ROOT/'models'/f'furry_glasses_lod_{tier}.glb';out.write_bytes(blob)
    return {'triangles':len(mesh.faces),'vertices':len(mesh.vertices),'drawCalls':1,'bytes':len(blob)}

if __name__=='__main__':
    for tier,n in LODS.items(): print(tier,build(tier,n))
