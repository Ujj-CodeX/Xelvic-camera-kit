#!/usr/bin/env python3
"""Strict validator for the Xelvic sunflower_forehead schemaVersion-1 package."""
from __future__ import annotations
import argparse, hashlib, json, math, struct, sys
from pathlib import Path
try:
    import jsonschema
except ImportError:
    jsonschema = None

def read_glb(path: Path):
    raw=path.read_bytes()
    if len(raw)<20: raise ValueError("GLB too small")
    magic,version,total=struct.unpack_from('<4sII',raw,0)
    if magic!=b'glTF' or version!=2 or total!=len(raw): raise ValueError("invalid GLB header/version/length")
    off=12; chunks=[]
    while off<len(raw):
        if off+8>len(raw): raise ValueError("truncated chunk header")
        ln,kind=struct.unpack_from('<I4s',raw,off); off+=8
        if off+ln>len(raw): raise ValueError("truncated GLB chunk")
        chunks.append((kind,raw[off:off+ln])); off+=ln
    j=next((b for k,b in chunks if k==b'JSON'),None)
    binary=next((b for k,b in chunks if k==bytes((66,73,78,0))),None)
    if j is None or binary is None: raise ValueError("GLB must contain JSON and BIN chunks")
    return json.loads(j.decode('utf-8').rstrip(' ')),binary,raw

def accessor_values(g,binary,index):
    a=g['accessors'][index]; v=g['bufferViews'][a['bufferView']]
    ct=a['componentType']; typ=a['type']
    fmt={5126:'f',5123:'H',5125:'I'}[ct]; width={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[typ]
    item=struct.calcsize('<'+fmt); stride=v.get('byteStride',item*width)
    start=v.get('byteOffset',0)+a.get('byteOffset',0); out=[]
    for n in range(a['count']):
        vals=struct.unpack_from('<'+fmt*width,binary,start+n*stride)
        out.append(vals[0] if width==1 else vals)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('package_dir',nargs='?',default='.')
    root=Path(ap.parse_args().package_dir).resolve()
    m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    s=json.loads((root/'schema/xfilter.schema.json').read_text(encoding='utf-8'))
    if jsonschema is not None:
        jsonschema.Draft202012Validator(s).validate(m)
    else:
        req_keys=set(s.get('required',[]))
        allowed_keys=set(s.get('properties',{}).keys())
        if req_keys-set(m.keys()): raise ValueError(f"missing required fields: {sorted(req_keys-set(m.keys()))}")
        if set(m.keys())-allowed_keys: raise ValueError(f"unknown fields: {sorted(set(m.keys())-allowed_keys)}")
    if not (root/'preview.jpg').is_file(): raise ValueError('missing preview.jpg')
    for a in m['assets']:
        p=root/a['path']; data=p.read_bytes()
        if len(data)!=a['bytes']: raise ValueError(f"byte count mismatch: {a['path']}")
        if hashlib.sha256(data).hexdigest()!=a['sha256']: raise ValueError(f"SHA-256 mismatch: {a['path']}")
    g,blob,_=read_glb(root/m['assets'][0]['path'])
    if g['asset']['version']!='2.0': raise ValueError('not glTF 2.0')
    if len(g.get('images',[]))!=3: raise ValueError('expected exactly 3 embedded textures')
    for image in g['images']:
        if 'bufferView' not in image or 'uri' in image or image.get('mimeType')!='image/png': raise ValueError(f"image is not an embedded PNG: {image.get('name')}")
        view=g['bufferViews'][image['bufferView']]
        if view['byteOffset']+view['byteLength']>len(blob): raise ValueError('embedded image extends outside BIN chunk')
    if len(g.get('animations',[]))!=3: raise ValueError('expected 3 animations')
    clip_names={a['name'] for a in g['animations']}
    if clip_names!=set(m['stats']['clips']): raise ValueError('GLB animations and manifest stats.clips differ')
    nodes={n['name']:i for i,n in enumerate(g['nodes'])}
    required={'sunflower_root','sunflower_pivot'}|set(m['model']['meshNames'])
    if required-set(nodes): raise ValueError(f"missing nodes: {sorted(required-set(nodes))}")
    if g['scenes'][0]['nodes']!=[nodes['sunflower_root']]: raise ValueError('unexpected scene root')
    if g['nodes'][nodes['sunflower_root']].get('children')!=[nodes['sunflower_pivot']]: raise ValueError('root/pivot hierarchy mismatch')
    pivot=g['nodes'][nodes['sunflower_pivot']]
    if pivot.get('scale',[1,1,1])!=[1,1,1] or pivot.get('rotation',[0,0,0,1])!=[0,0,0,1]: raise ValueError('animated pivot must rest at identity scale/rotation')
    for mat in g['materials']:
        pbr=mat['pbrMetallicRoughness']
        if set(pbr)-{'baseColorTexture','metallicFactor','roughnessFactor'}: raise ValueError('material has a property outside the baseColorTexture-only contract')
        if 'baseColorTexture' not in pbr or pbr.get('metallicFactor')!=0 or pbr.get('roughnessFactor')!=1: raise ValueError('material violates base-color texture / Lambert factors')
        if mat.get('alphaMode')!='OPAQUE' or mat.get('doubleSided') is not True: raise ValueError('material alpha/double-sided contract mismatch')
    # Check the sampler values match the package contract.
    samp=g['samplers'][0]
    if samp!={'magFilter':9729,'minFilter':9987,'wrapS':10497,'wrapT':10497}: raise ValueError('GLB sampler differs from manifest contract')
    # Animation targets, timing and endpoints.
    clips={x['id']:x for x in m['behaviors']['clips']}
    for anim in g['animations']:
        spec=clips[anim['name']]
        for ch in anim['channels']:
            if ch['target']['node']!=nodes['sunflower_pivot']: raise ValueError(f"{anim['name']} targets a non-pivot node")
        for samp in anim['samplers']:
            times=accessor_values(g,blob,samp['input'])
            if abs(float(times[-1])-float(spec['durationSec']))>1e-5: raise ValueError(f"{anim['name']} duration differs from manifest")
        if anim['name'] in ('bloom_in','burst'):
            values=accessor_values(g,blob,anim['samplers'][0]['output'])
            if any(abs(float(x)-1.0)>1e-5 for x in values[-1]): raise ValueError(f"{anim['name']} scale clip does not end at 1")
        if anim['name']=='idle_sway':
            vals=accessor_values(g,blob,anim['samplers'][0]['output'])
            if any(abs(float(a)-float(b))>1e-5 for a,b in zip(vals[0],vals[-1])): raise ValueError('idle_sway rotation loop does not close')
    # Tier node sets must exactly partition the available LOD mesh nodes.
    mesh_nodes=set(m['model']['meshNames'])
    for tier in m['tiers']:
        vis=set(tier['visibleNodes']); hid=set(tier['hideNodes'])
        if vis&hid or vis|hid!=mesh_nodes: raise ValueError(f"tier {tier['id']} does not partition all LOD nodes")
        if len(vis)!=tier['drawCalls']: raise ValueError(f"tier {tier['id']} draw calls differ from visible mesh-node count")
        total=0
        for name in vis:
            node=g['nodes'][nodes[name]]; mesh=g['meshes'][node['mesh']]
            if len(mesh['primitives'])!=1: raise ValueError('each LOD node must draw one primitive')
            prim=mesh['primitives'][0]
            if prim['mode']!=4: raise ValueError('primitive must be TRIANGLES')
            idx=g['accessors'][prim['indices']]
            if idx['componentType']!=5123: raise ValueError('indices must be uint16')
            total+=idx['count']//3
        if total!=tier['triangles']: raise ValueError(f"tier {tier['id']} triangles: manifest={tier['triangles']}, actual={total}")
    calculated={t['id']:t['triangles'] for t in m['tiers']}
    if calculated!=m['stats']['triangles']: raise ValueError('stats.triangles does not match tier geometry')
    if {t['id']:t['drawCalls'] for t in m['tiers']}!=m['stats']['drawCalls']: raise ValueError('stats.drawCalls does not match tiers')
    if len(m['assets'])!=1 or m['assets'][0]['id']!=m['model']['asset']: raise ValueError('manifest asset/model asset mismatch')
    # Rest clearance against head-proxy ellipsoid, using Tier A's exact geometry.
    CANONICAL_LANDMARKS_FW={
        297:(-0.229935,0.478393,0.402188),
        338:(-0.123366,0.493391,0.437807),
        332:(-0.334869,0.444449,0.332450),
        10:(0.000000,0.495089,0.451314),
        333:(-0.296117,0.368393,0.399187),
    }
    anchor_name=m['layer'].get('anchor','head_center')
    layer_off=m['layer']['offsetFW']; scale=m['layer']['scale']; radii=m['occlusion']['headProxy']['radiiFW']
    if anchor_name=='head_center':
        offset=layer_off
    else:
        a_def=m['anchors'][anchor_name]
        lms=a_def.get('landmarks',[])
        w_sum=sum(float(x['weight']) for x in lms)
        if abs(w_sum-1.0)>1e-5: raise ValueError(f"anchor {anchor_name} weights must sum to 1.0, got {w_sum}")
        lm_fw=[sum(float(x['weight'])*CANONICAL_LANDMARKS_FW[int(x['id'])][i] for x in lms) for i in range(3)]
        a_off=a_def.get('offsetFW',[0.0,0.0,0.0])
        hc_off=m['anchors']['head_center']['offsetFW']
        hc_eff=[hc_off[0],hc_off[1],hc_off[2]-0.45*radii[2]]
        offset=[lm_fw[i]+float(a_off[i])+float(layer_off[i])-float(hc_eff[i]) for i in range(3)]
    min_q=float('inf')
    for name in next(t for t in m['tiers'] if t['id']=='A')['visibleNodes']:
        node=g['nodes'][nodes[name]]; prim=g['meshes'][node['mesh']]['primitives'][0]
        for p in accessor_values(g,blob,prim['attributes']['POSITION']):
            q=sum(((offset[i]+scale*float(p[i]))/radii[i])**2 for i in range(3)); min_q=min(min_q,q)
    clearance=math.sqrt(min_q); threshold=m['occlusion']['headProxy']['clearanceRatioMin']
    if clearance+1e-6<threshold: raise ValueError(f"head-proxy clearance {clearance:.4f} below minimum {threshold:.2f}")
    if abs(clearance-m['occlusion']['headProxy']['measuredRestClearanceRatio'])>2e-4: raise ValueError('manifest measured clearance does not match GLB geometry')
    print('PASS: strict JSON Schema (unknown fields rejected)')
    print('PASS: asset byte count and SHA-256')
    print('PASS: glTF 2.0, embedded PNG textures, node hierarchy, samplers and materials')
    print('PASS: animation names, targets, durations, closed loop and scale endpoints')
    print('PASS: LOD node sets, draw calls and indexed triangle budgets')
    print(f'PASS: head-proxy clearance {clearance:.4f} >= {threshold:.2f}')
    print(f"PACKAGE: {m['id']} | { (root/m['assets'][0]['path']).stat().st_size:,} bytes")
if __name__=='__main__':
    try: main()
    except Exception as e:
        print('FAIL:',e,file=sys.stderr); sys.exit(1)
