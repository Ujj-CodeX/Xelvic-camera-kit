from pathlib import Path
import json, hashlib, zipfile, struct, math, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import trimesh
from shapely.geometry import Polygon
from shapely.ops import triangulate

ROOT = Path('/mnt/data/Heart_Mood_Filter_RnD')
if ROOT.exists(): shutil.rmtree(ROOT)
(ROOT/'models').mkdir(parents=True)
(ROOT/'assets').mkdir()
(ROOT/'source').mkdir()
(ROOT/'validation').mkdir()

# Reference composition: small, glossy pink hearts in a gentle arc following the upper hairline.
HEARTS = [
    (-0.430, -0.064, 0.073, -12.0),
    (-0.322, -0.022, 0.078, -8.0),
    (-0.215,  0.025, 0.087, -5.0),
    (-0.108,  0.063, 0.079, -2.0),
    ( 0.000,  0.082, 0.101,  2.0),
    ( 0.108,  0.064, 0.079,  5.0),
    ( 0.215,  0.025, 0.087,  8.0),
    ( 0.322, -0.022, 0.078, 10.0),
    ( 0.430, -0.064, 0.073, 13.0),
]

# High-quality 5x2 atlas: one warm pastel pink variant per heart, opaque sRGB.
TILE = 256
COLS, ROWS = 5, 2
atlas = Image.new('RGBA', (TILE*COLS, TILE*ROWS), (255,255,255,255))
base_cols = [
    (255,190,213), (255,168,202), (255,184,216), (255,151,192), (255,197,220),
    (255,164,202), (255,185,214), (255,158,198), (255,193,220)
]
for i, base in enumerate(base_cols):
    tile = Image.new('RGBA',(TILE,TILE),(255,255,255,255))
    px = tile.load()
    for y in range(TILE):
        for x in range(TILE):
            # Subtle radial gloss and lower-right rose shading, no alpha holes.
            nx=(x-TILE*0.34)/(TILE*0.8); ny=(y-TILE*0.24)/(TILE*0.85)
            d=min(1.0, math.sqrt(nx*nx+ny*ny))
            shade=1.05-0.22*d-0.045*(y/TILE)
            r,g,b=base
            px[x,y]=(int(max(0,min(255,r*shade))),int(max(0,min(255,g*shade))),int(max(0,min(255,b*shade))),255)
    # Soft specular highlight on front upper lobe.
    hi=Image.new('RGBA',(TILE,TILE),(0,0,0,0)); hd=ImageDraw.Draw(hi)
    hd.ellipse((TILE*.19,TILE*.14,TILE*.47,TILE*.35), fill=(255,255,255,85))
    hi=hi.filter(ImageFilter.GaussianBlur(12))
    tile=Image.alpha_composite(tile,hi)
    atlas.alpha_composite(tile, ((i%COLS)*TILE,(i//COLS)*TILE))
atlas_path=ROOT/'assets'/'heart_mood_atlas.png'
atlas.save(atlas_path, optimize=True)

# Thumbnail card matching the reference composition, without using the person's photo.
thumb=Image.new('RGBA',(720,420),(255,247,250,255))
# faint airy background gradient
p=thumb.load()
for y in range(420):
    for x in range(720):
        t=y/419
        p[x,y]=(255, int(250-6*t), int(252-3*t), 255)
# soft pale halo arch
halo=Image.new('RGBA',thumb.size,(0,0,0,0)); draw=ImageDraw.Draw(halo)
draw.arc((82,30,638,440), 190, 350, fill=(255,190,216,90), width=18)
halo=halo.filter(ImageFilter.GaussianBlur(16)); thumb=Image.alpha_composite(thumb,halo)

def heart_poly(cx,cy,w,h,rot=0):
    pts=[]
    for t in np.linspace(0,2*math.pi,120,endpoint=False):
        x=16*math.sin(t)**3/32
        y=(13*math.cos(t)-5*math.cos(2*t)-2*math.cos(3*t)-math.cos(4*t))/32
        x*=w; y*=h
        a=math.radians(rot)
        xr=x*math.cos(a)-y*math.sin(a); yr=x*math.sin(a)+y*math.cos(a)
        pts.append((cx+xr,cy-yr))
    return pts
for i,(x,y,w,rot) in enumerate(HEARTS):
    cx=360+x*600; cy=148-y*520
    ww=w*600; hh=ww*.90
    poly=heart_poly(cx,cy,ww,hh,rot)
    # soft halo then pink body
    glow=Image.new('RGBA',thumb.size,(0,0,0,0)); gd=ImageDraw.Draw(glow)
    gd.polygon(poly,fill=(255,135,188,100)); glow=glow.filter(ImageFilter.GaussianBlur(13))
    thumb=Image.alpha_composite(thumb,glow)
    hd=ImageDraw.Draw(thumb)
    hd.polygon(poly,fill=(255,165+(i%3)*7,202+(i%2)*9,255),outline=(255,130,181,255))
    # tiny white glossy glint
    hd.ellipse((cx-ww*.20,cy-hh*.27,cx-ww*.03,cy-hh*.13),fill=(255,255,255,190))
# typography
D=ImageDraw.Draw(thumb)
D.text((34,348),'HEART MOOD FILTER',fill=(154,81,115,255),stroke_width=0)
D.text((34,375),'Floating pastel heart crown  •  Mesh-first 3D prop',fill=(175,122,147,255))
thumb_path=ROOT/'thumbnail.png'; thumb.save(thumb_path,optimize=True)

# Create bevelled, extruded heart geometry. Front is +Z; +Y is up.
def make_heart_mesh(segment_count, center, width, rotation_deg, atlas_cell, zcenter):
    # Parametric heart outline, normalized to width 1 and height ~0.91.
    ts=np.linspace(0,2*math.pi,segment_count,endpoint=False)
    raw=[]
    for t in ts:
        x=16*(math.sin(t)**3)
        y=13*math.cos(t)-5*math.cos(2*t)-2*math.cos(3*t)-math.cos(4*t)
        raw.append((x/32.0,y/32.0))
    raw=np.array(raw,dtype=float)
    raw[:,0] *= width
    raw[:,1] *= width
    a=math.radians(rotation_deg)
    rot=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
    raw=raw@rot.T
    raw[:,0]+=center[0]; raw[:,1]+=center[1]
    raw[:,1]*=0.90
    poly=Polygon(raw)
    if not poly.is_valid: poly=poly.buffer(0)
    minx,miny,maxx,maxy=poly.bounds
    # Atlas cell rectangle; retain a small inset to avoid sampling neighboring cells.
    col=atlas_cell%COLS; row=atlas_cell//COLS
    u0=(col*TILE+10)/(TILE*COLS); u1=((col+1)*TILE-10)/(TILE*COLS)
    v0=1.0-((row+1)*TILE-10)/(TILE*ROWS); v1=1.0-(row*TILE+10)/(TILE*ROWS)
    def uv(x,y):
        uu=u0+(x-minx)/(maxx-minx)*(u1-u0)
        vv=v0+(y-miny)/(maxy-miny)*(v1-v0)
        return (uu,vv)
    verts=[]; uvs=[]; faces=[]
    depth=width*0.17; bevel=width*0.045; outerScale=1.0; innerScale=0.88
    # local outline is recentered at each heart centre for bevel ring construction.
    local=raw.copy(); local[:,0]-=center[0]; local[:,1]-=center[1]
    # front and back triangulated inset polygon
    inset=local*innerScale
    poly_in=Polygon(inset)
    tris=[t for t in triangulate(poly_in) if poly_in.covers(t.representative_point())]
    # Add face triangles separately to preserve robust UVs and material layout.
    for tri in tris:
        coords=list(tri.exterior.coords)[:3]
        area=sum(coords[j][0]*coords[(j+1)%3][1]-coords[(j+1)%3][0]*coords[j][1] for j in range(3))
        if area<0: coords=coords[::-1]
        base=len(verts)
        for x,y in coords:
            xx=x+center[0]; yy=y+center[1]
            verts.append((xx,yy,zcenter+depth/2))
            uvs.append(uv(xx,yy))
        faces.append((base,base+1,base+2))
        # Backface reverse winding
        base=len(verts)
        for x,y in coords:
            xx=x+center[0]; yy=y+center[1]
            verts.append((xx,yy,zcenter-depth/2))
            uvs.append(uv(xx,yy))
        faces.append((base,base+2,base+1))
    n=len(local)
    # Bevel rings: front outer -> front inset; back inset -> back outer.
    for i in range(n):
        j=(i+1)%n
        # Front bevel outer / inner
        p0=local[i]*outerScale+np.array(center); p1=local[j]*outerScale+np.array(center)
        q0=local[i]*innerScale+np.array(center); q1=local[j]*innerScale+np.array(center)
        zfo=zcenter+depth/2-bevel*0.35; zfi=zcenter+depth/2
        base=len(verts)
        for pt,z in [(p0,zfo),(p1,zfo),(q1,zfi),(q0,zfi)]:
            verts.append((pt[0],pt[1],z)); uvs.append(uv(pt[0],pt[1]))
        faces.extend([(base,base+1,base+2),(base,base+2,base+3)])
        # Back bevel
        zbo=zcenter-depth/2+bevel*0.35; zbi=zcenter-depth/2
        base=len(verts)
        for pt,z in [(q0,zbi),(q1,zbi),(p1,zbo),(p0,zbo)]:
            verts.append((pt[0],pt[1],z)); uvs.append(uv(pt[0],pt[1]))
        faces.extend([(base,base+1,base+2),(base,base+2,base+3)])
        # Side wall
        base=len(verts)
        for pt,z in [(p0,zfo),(p1,zfo),(p1,zbo),(p0,zbo)]:
            verts.append((pt[0],pt[1],z)); uvs.append(uv(pt[0],pt[1]))
        faces.extend([(base,base+1,base+2),(base,base+2,base+3)])
    return np.asarray(verts,dtype=np.float32), np.asarray(faces,dtype=np.uint32), np.asarray(uvs,dtype=np.float32)

# Trimesh currently emits UNSIGNED_INT index accessors even for small meshes.
# Convert the GLB index buffer to UNSIGNED_SHORT when the maximum index fits.
def force_indices_uint16(blob):
    magic, version, total = struct.unpack_from("<4sII", blob, 0)
    assert magic == b"glTF" and version == 2 and total == len(blob)
    json_len, json_type = struct.unpack_from("<II", blob, 12)
    assert json_type == 0x4E4F534A
    json_bytes = blob[20:20+json_len]
    gltf = json.loads(json_bytes.decode("utf-8").rstrip(" \t\r\n\0"))
    bin_header = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<II", blob, bin_header)
    assert bin_type == 0x004E4942
    binary = bytearray(blob[bin_header+8:bin_header+8+bin_len])
    primitive = gltf["meshes"][0]["primitives"][0]
    accessor = gltf["accessors"][primitive["indices"]]
    if accessor["componentType"] == 5123:
        return blob
    assert accessor["componentType"] == 5125
    view_index = accessor["bufferView"]
    view = gltf["bufferViews"][view_index]
    old_offset = view.get("byteOffset", 0)
    old_length = view["byteLength"]
    count = accessor["count"]
    raw = np.frombuffer(binary[old_offset:old_offset+old_length], dtype="<u4", count=count)
    assert int(raw.max()) <= 65535
    packed = raw.astype("<u2").tobytes()
    delta = old_length - len(packed)
    new_binary = bytearray(binary[:old_offset] + packed + binary[old_offset+old_length:])
    view["byteLength"] = len(packed)
    accessor["componentType"] = 5123
    # All following bufferViews point to data moved earlier by the index-buffer size delta.
    for index, other in enumerate(gltf["bufferViews"]):
        if index != view_index and other.get("byteOffset", 0) >= old_offset + old_length:
            other["byteOffset"] -= delta
    gltf["buffers"][0]["byteLength"] = len(new_binary)
    # Rebuild a valid GLB with four-byte-aligned JSON and BIN chunks.
    json_out = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_out += b" " * ((4 - len(json_out) % 4) % 4)
    new_binary += b"\0" * ((4 - len(new_binary) % 4) % 4)
    total_len = 12 + 8 + len(json_out) + 8 + len(new_binary)
    return (struct.pack("<4sII", b"glTF", 2, total_len) +
            struct.pack("<II", len(json_out), 0x4E4F534A) + json_out +
            struct.pack("<II", len(new_binary), 0x004E4942) + new_binary)

# The crown wraps subtly around the head: side hearts sit slightly farther back in Z.
def build_lod(seg_count, path):
    all_v=[]; all_f=[]; all_uv=[]; offset=0
    for idx,(x,y,w,rot) in enumerate(HEARTS):
        z=-0.105*(abs(x)/0.43)**1.75 + 0.012*y
        v,f,uv=make_heart_mesh(seg_count,(x,y),w,rot,idx,z)
        all_v.append(v); all_f.append(f+offset); all_uv.append(uv); offset+=len(v)
    vertices=np.vstack(all_v); faces=np.vstack(all_f); uvs=np.vstack(all_uv)
    # Center the group pivot at its bounding-box centre; the manifest supplies the head-centre offset.
    bounds=np.vstack([vertices.min(axis=0),vertices.max(axis=0)])
    pivot=(bounds[0]+bounds[1])/2
    vertices-=pivot
    # Single mesh / single material / one draw call. Preserve vertex data and disable process welding.
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    mesh.visual=trimesh.visual.texture.TextureVisuals(uv=uvs, image=atlas)
    mat=trimesh.visual.material.PBRMaterial(
        name='HeartMood_SRGB_PastelPink',
        baseColorFactor=[255,255,255,255],
        baseColorTexture=atlas,
        metallicFactor=0.02,
        roughnessFactor=0.31,
        alphaMode='OPAQUE',
        doubleSided=True
    )
    mesh.visual.material=mat
    scene=trimesh.Scene()
    scene.add_geometry(mesh,geom_name='heart_mood_crown',node_name='HeartMood_Crown_Pivot')
    blob=trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    blob=force_indices_uint16(blob)
    Path(path).write_bytes(blob)
    return {
      'triangles':int(len(faces)), 'vertices':int(len(vertices)), 'drawCalls':1,
      'boundsFW':(vertices.max(axis=0)-vertices.min(axis=0)).round(6).tolist(),
      'pivotOriginalFW':pivot.round(6).tolist(), 'bytes':len(blob)
    }

metrics={}
for tier,seg in [('A',64),('B',40),('C',24)]:
    metrics[tier]=build_lod(seg,ROOT/'models'/f'heart_mood_crown_lod_{tier.lower()}.glb')

# Build manifest. All tunable numeric placement/occlusion/behaviour defaults are mirrored in calibration paths.
calibration_fields=[]
def tune(path, default, minimum, maximum, unit):
    calibration_fields.append({'path':path,'default':default,'min':minimum,'max':maximum,'unit':unit})

tunables=[
 ('layers.heart_crown.transform.offsetFW.x',0.0,-0.05,0.05,'FW'),
 ('layers.heart_crown.transform.offsetFW.y',0.56,0.48,0.68,'FW'),
 ('layers.heart_crown.transform.offsetFW.z',0.18,0.12,0.24,'FW'),
 ('layers.heart_crown.transform.scaleFW',1.0,0.90,1.10,'multiplier'),
 ('layers.heart_crown.occlusion.faceMesh.pushBackFW',0.015,0.005,0.025,'FW'),
 ('layers.heart_crown.occlusion.headProxyEllipsoid.radiiFW.x',0.62,0.56,0.68,'FW'),
 ('layers.heart_crown.occlusion.headProxyEllipsoid.radiiFW.y',0.82,0.74,0.90,'FW'),
 ('layers.heart_crown.occlusion.headProxyEllipsoid.radiiFW.z',0.35,0.30,0.40,'FW'),
 ('layers.heart_crown.behaviors.gentleFloat.maxAngleDeg',2.2,0.0,3.0,'deg'),
 ('layers.heart_crown.behaviors.gentleFloat.maxTranslationFW',0.006,0.0,0.010,'FW'),
 ('layers.heart_crown.behaviors.gentleFloat.periodMs',3200,2400,4200,'ms'),
 ('layers.heart_crown.lifecycle.faceLost.holdMs',160,100,240,'ms'),
 ('layers.heart_crown.lifecycle.faceLost.fadeMs',220,160,320,'ms'),
 ('layers.heart_crown.transform.rotationDeg.x',0.0,-5.0,5.0,'deg'),
 ('layers.heart_crown.transform.rotationDeg.y',0.0,-5.0,5.0,'deg'),
 ('layers.heart_crown.transform.rotationDeg.z',0.0,-5.0,5.0,'deg'),
 ('layers.heart_crown.fit.ratio',1.0,0.90,1.10,'multiplier'),
 ('layers.heart_crown.occlusion.headProxyEllipsoid.centerOffsetFW.x',0.0,-0.05,0.05,'FW'),
 ('layers.heart_crown.occlusion.headProxyEllipsoid.centerOffsetFW.y',0.0,-0.05,0.05,'FW'),
 ('layers.heart_crown.occlusion.headProxyEllipsoid.centerOffsetFW.z',0.0,-0.05,0.05,'FW'),
 ('layers.heart_crown.material.metallicFactor',0.02,0.0,0.08,'factor'),
 ('layers.heart_crown.material.roughnessFactor',0.31,0.24,0.42,'factor'),
]
for t in tunables: tune(*t)
for idx,(hx,hy,hw,hrot) in enumerate(HEARTS):
    hz=-0.105*(abs(hx)/0.43)**1.75 + 0.012*hy
    prefix=f'layers.heart_crown.geometryDesign.heartLayout.{idx}'
    tune(prefix+'.centerFW.x',round(hx,6),round(hx-0.015,6),round(hx+0.015,6),'FW')
    tune(prefix+'.centerFW.y',round(hy,6),round(hy-0.015,6),round(hy+0.015,6),'FW')
    tune(prefix+'.centerFW.z',round(hz,6),round(hz-0.01,6),round(hz+0.01,6),'FW')
    tune(prefix+'.widthFW',round(hw,6),round(hw*0.9,6),round(hw*1.1,6),'FW')
    tune(prefix+'.rotationDeg',round(hrot,6),round(hrot-3,6),round(hrot+3,6),'deg')

asset_records=[]
for p in [ROOT/'thumbnail.png', atlas_path, *sorted((ROOT/'models').glob('*.glb'))]:
    data=p.read_bytes()
    asset_records.append({'path':p.relative_to(ROOT).as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})

manifest={
 'packageFormat':'xelvic-filter-rnd',
 'schemaVersion':3,
 'id':'heart_mood_filter',
 'name':'Heart Mood Filter',
 'version':'1.0.0',
 'description':'A soft pastel pink heart crown floating along the upper hairline, inspired by the supplied reference image.',
 'minimumRuntimeVersion':3,
 'status':'RND_AUTHORING_PACKAGE_NOT_IMPORTABLE_BY_CURRENT_RUNTIME',
 'coordinateSpace':{
   'unit':'FACE_WIDTH',
   'axes':{'x':'+subject-right','y':'+up','z':'+camera'},
   'modelFront':'+Z','modelUp':'+Y','packageMirroring':'NONE',
   'note':'Engine applies front-camera display mirroring exactly once. No asset-side flips or side compensation.'
 },
 'requires':{'features':['model.gltf2@1','anchor.headCenter@1','occlusion.faceMesh@1','occlusion.headProxyEllipsoid@1','behavior.boundedFloat@1']},
 'compatibility':{'maxFaces':1,'faceIndex':0,'requiredBlendshapes':[],'minimumYawDeg':-30,'maximumYawDeg':30},
 'assets':asset_records,
 'layers':[{
   'id':'heart_crown','type':'MODEL_3D','renderStage':'OPAQUE_MODELS',
   'lods':[
      {'tier':'A','asset':'models/heart_mood_crown_lod_a.glb','drawCalls':metrics['A']['drawCalls'],'triangles':metrics['A']['triangles'],'heartCount':len(HEARTS)},
      {'tier':'B','asset':'models/heart_mood_crown_lod_b.glb','drawCalls':metrics['B']['drawCalls'],'triangles':metrics['B']['triangles'],'heartCount':len(HEARTS)},
      {'tier':'C','asset':'models/heart_mood_crown_lod_c.glb','drawCalls':metrics['C']['drawCalls'],'triangles':metrics['C']['triangles'],'heartCount':len(HEARTS)}
   ],
   'anchor':{'type':'head_center','index':0},
   'geometryDesign':{
      'unit':'FACE_WIDTH',
      'heartCount':len(HEARTS),
      'heartLayout':[{'id':f'heart_{i+1:02d}','centerFW':{'x':round(hx,6),'y':round(hy,6),'z':round(-0.105*(abs(hx)/0.43)**1.75 + 0.012*hy,6)},'widthFW':round(hw,6),'rotationDeg':round(hrot,6)} for i,(hx,hy,hw,hrot) in enumerate(HEARTS)],
      'lodSegments':{'A':64,'B':40,'C':24},
      'lodPolicy':'same heart layout and count; reduce contour tessellation only'
   },
   'transform':{
      'offsetFW':{'x':0.0,'y':0.56,'z':0.18},
      'scaleFW':1.0,
      'rotationDeg':{'x':0.0,'y':0.0,'z':0.0}
   },
   'fit':{'to':'faceWidth','ratio':1.0},
   'occlusion':{
      'faceMesh':{'enabled':True,'pushBackFW':0.015,'depthWrite':True,'colorWrite':False},
      'headProxyEllipsoid':{'enabled':True,'centerOffsetFW':{'x':0.0,'y':0.0,'z':0.0},'radiiFW':{'x':0.62,'y':0.82,'z':0.35},'depthWrite':True,'colorWrite':False},
      'earTempleProxy':{'enabled':False,'reason':'The crown remains within the upper forehead silhouette and does not require temple/ear occlusion.'}
   },
   'material':{'workflow':'PBR_METALLIC_ROUGHNESS','baseColorTexture':'EMBEDDED_SRGB','alphaMode':'OPAQUE','mipmaps':True,'metallicFactor':0.02,'roughnessFactor':0.31},
   'behaviors':{
     'gentleFloat':{'type':'BOUNDED_FLOAT','maxAngleDeg':2.2,'maxTranslationFW':0.006,'periodMs':3200,'restPose':{'translationFW':[0.0,0.0,0.0],'rotationQuat':[0.0,0.0,0.0,1.0],'scale':[1.0,1.0,1.0]}},
     'tierPolicy':{'A':'enabled','B':'enabled','C':'enabled','lodOnlyReduces':'meshDetail','allHeartComponentsPreserved':True}
   },
   'lifecycle':{'faceLost':{'holdMs':160,'fadeMs':220,'then':'RESET'},'filterSwitch':{'trackerRestart':False,'reuseTrackingState':True,'preloadTexture':True}}
 }],
 'calibration':{'tunableFields':calibration_fields},
 'renderPlan':{'passOrder':['CAMERA_FRAME','REGION_MASK','CAMERA_PIXEL_PASSES','CLEAR_DEPTH_AND_OCCLUDERS','OPAQUE_MODELS','ALPHA_LENSES','POST_NODES','OVERLAY_NODES'],'layerStage':'OPAQUE_MODELS','facePaintBeforeProps':True,'postPassCount':1},
 'performance':{'drawCallsPerTier':{k:metrics[k]['drawCalls'] for k in metrics},'triangleCountsPerTier':{k:metrics[k]['triangles'] for k in metrics},'texturePolicy':'embedded atlas, decode/cache once, GL upload only on GL thread','cachePolicy':'LRU and bounded by engine texture budget'},
 'verification':{'staticAssetChecks':'GENERATED_AND_CHECKED','physicalDeviceChecks':'NOT_RUN','runtimeImport':'NOT_SUPPORTED_BY_CURRENT_FILTER_PACKAGEMANAGER'}
}
manifest_path=ROOT/'manifest.json'; manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')

# Authoring README communicates runtime compatibility and how values map to the user's checklist.
readme='''# Heart Mood Filter — R&D Authoring Package\n\nA pastel pink floating-heart crown matching the supplied reference composition. This package is authored for the planned Xelvic mesh/3D prop pipeline and aligned with the attached revised implementation plan.\n\n## Important runtime compatibility\n\nThis is an **R&D authoring ZIP**, not a currently importable `.xfilter` for the existing app. The current repository `FilterPackageManager` allows `.json`, `.png`, and `.webp` archive entries only; GLB is rejected, and the active camera pipeline does not yet render 3D models. The package uses schemaVersion 3 / minimumRuntimeVersion 3 as a proposed model-prop contract. Do not rename this ZIP to `.xfilter` or import it into the current runtime until the package validator, model loader, renderer router, and GLB support are implemented.\n\n## Contents\n\n- `manifest.json` — effect definition, FACE_WIDTH coordinates, head-centre anchor, calibration tunables, face/head occlusion, bounded behaviour, lifecycle, tiers and metrics.\n- `models/heart_mood_crown_lod_a.glb` — high-detail model.\n- `models/heart_mood_crown_lod_b.glb` — medium detail, all nine hearts retained.\n- `models/heart_mood_crown_lod_c.glb` — low detail, all nine hearts retained.\n- `assets/heart_mood_atlas.png` — source preview of the embedded sRGB atlas; each GLB embeds the texture.\n- `thumbnail.png` — package thumbnail.\n- `validation/report.json` — hashes, GLB attributes and metric checks.\n- `source/build_heart_mood_filter.py` — reproducible geometry/texture generator.\n\n## Design and alignment\n\n- 9 small hearts arranged in a gentle upper-hairline arc. The full crown spans about 0.93 face-width (FW); individual heart width varies around 0.073–0.101 FW to match the reference's delicate floating hearts. Each baked heart centre, width and rotation is also documented as a calibration field so the source mesh can be regenerated consistently.\n- Model axes: +X subject-right, +Y up, +Z camera. No mirror/flip/side compensation is baked into the assets.\n- Each LOD uses one combined mesh and one material (one draw call). LOD B/C reduce contour tessellation only; all nine hearts remain present.\n- Model pivot is the centre of the combined model bounds; rest scale is 1 and rest rotation is identity. Layer placement is calibrated from `head_center` using FW units.\n- Face-mesh occluder plus a head-proxy ellipsoid handle face/head depth. Ear/temple proxy is deliberately disabled because this effect remains within the upper forehead/hairline region.\n- The model renders in Stage 5 `OPAQUE_MODELS`, after Stage 4 `CLEAR_DEPTH_AND_OCCLUDERS`, and before transparent props/post/overlay passes. No camera-pixel paint or eye-colour pass is required.\n- Gentle float is bounded (2.2 degrees; 0.006 FW), rest pose returns to identity, and face loss uses hold → fade → reset.\n- No blendshapes are required. Compatibility is one face, index 0.\n\n## Calibration\n\nEvery placement, occlusion and behaviour tuning default is mirrored by a path in `calibration.tunableFields`. Tune using the intended engine's canonical head centre and face-width measurement; do not add package-side mirroring. The initial values are authoring estimates and must be validated with frontal, ±30-degree yaw, roll, near/far, glasses, low-light, pause/resume and rapid filter-switch tests.\n\n## GLB technical contract\n\n- glTF 2.0 binary (GLB).\n- Vertex attributes: POSITION, NORMAL, TEXCOORD_0.\n- Index accessor: UNSIGNED_SHORT.\n- Single opaque sRGB base-colour texture, embedded in every GLB; enable mipmap generation at runtime.\n- Front +Z, up +Y; centered pivot; rest scale 1 and identity rotation.\n- Actual triangle and draw-call counts are recorded in manifest and validation report.\n\n## Render/occlusion requirements\n\nThe renderer must draw the face mesh and head proxy in a depth-only occluder pass before opaque models. Both occluders and the heart model must use the same camera projection/depth convention. The `pushBackFW` value is a small calibration bias; if the renderer uses a different depth convention, it must map this value consistently rather than applying an arbitrary screen-space offset.\n\n## Reproducibility\n\nRun the generator with Python packages `numpy`, `Pillow`, `shapely`, and `trimesh`. The generator rebuilds the three LOD GLBs, thumbnail, atlas, manifest, and validation report.\n\n## Verification status\n\nStatic file structure, hashes, GLB container fields, vertex attributes, index widths, and declared triangle/draw-call counts are validated by the included report. No Android runtime import or physical-device visual verification is claimed.\n'''
(ROOT/'README.md').write_text(readme,encoding='utf-8')

# Copy generator into package after writing it separately at /mnt/data.
script_src=Path('/mnt/data/build_heart_mood.py')
shutil.copy2(script_src,ROOT/'source'/'build_heart_mood_filter.py')

# Parse GLB chunks and perform structural assertions.
def parse_glb(path):
    b=Path(path).read_bytes()
    magic,version,total=struct.unpack_from('<4sII',b,0)
    assert magic==b'glTF' and version==2 and total==len(b)
    json_len,json_type=struct.unpack_from('<II',b,12)
    assert json_type==0x4E4F534A
    gltf=json.loads(b[20:20+json_len].decode('utf-8').rstrip(' \t\r\n\0'))
    return gltf,b
checks=[]
for tier in ['A','B','C']:
    path=ROOT/'models'/f'heart_mood_crown_lod_{tier.lower()}.glb'
    gltf,b=parse_glb(path)
    primitive=gltf['meshes'][0]['primitives'][0]
    attrs=sorted(primitive['attributes'].keys())
    idx_accessor=gltf['accessors'][primitive['indices']]
    idx_type=idx_accessor['componentType']
    assert attrs==['NORMAL','POSITION','TEXCOORD_0'], (tier,attrs)
    assert idx_type==5123, (tier,idx_type)
    assert 'images' in gltf and 'bufferView' in gltf['images'][0], 'Texture must be embedded'
    assert len(gltf.get('meshes',[]))==1 and len(gltf['meshes'][0]['primitives'])==1, 'One mesh primitive expected'
    assert primitive.get('mode',4)==4
    count=idx_accessor['count']//3
    assert count==metrics[tier]['triangles'], (tier,count,metrics[tier]['triangles'])
    checks.append({'tier':tier,'glbVersion':gltf['asset']['version'],'attributes':attrs,'indicesComponentType':'UNSIGNED_SHORT','indexCount':idx_accessor['count'],'triangles':count,'drawCalls':1,'embeddedTexture':True,'bytes':len(b)})

# Validate calibration field defaults sync to layer values.
layer=manifest['layers'][0]
lookup={t['path']:t['default'] for t in manifest['calibration']['tunableFields']}
checks_sync={
 'offset_x':lookup['layers.heart_crown.transform.offsetFW.x']==layer['transform']['offsetFW']['x'],
 'offset_y':lookup['layers.heart_crown.transform.offsetFW.y']==layer['transform']['offsetFW']['y'],
 'offset_z':lookup['layers.heart_crown.transform.offsetFW.z']==layer['transform']['offsetFW']['z'],
 'scale':lookup['layers.heart_crown.transform.scaleFW']==layer['transform']['scaleFW'],
 'face_pushback':lookup['layers.heart_crown.occlusion.faceMesh.pushBackFW']==layer['occlusion']['faceMesh']['pushBackFW'],
 'ellipsoid_rx':lookup['layers.heart_crown.occlusion.headProxyEllipsoid.radiiFW.x']==layer['occlusion']['headProxyEllipsoid']['radiiFW']['x'],
 'ellipsoid_ry':lookup['layers.heart_crown.occlusion.headProxyEllipsoid.radiiFW.y']==layer['occlusion']['headProxyEllipsoid']['radiiFW']['y'],
 'ellipsoid_rz':lookup['layers.heart_crown.occlusion.headProxyEllipsoid.radiiFW.z']==layer['occlusion']['headProxyEllipsoid']['radiiFW']['z'],
 'float_angle':lookup['layers.heart_crown.behaviors.gentleFloat.maxAngleDeg']==layer['behaviors']['gentleFloat']['maxAngleDeg'],
 'float_translation':lookup['layers.heart_crown.behaviors.gentleFloat.maxTranslationFW']==layer['behaviors']['gentleFloat']['maxTranslationFW'],
 'float_period':lookup['layers.heart_crown.behaviors.gentleFloat.periodMs']==layer['behaviors']['gentleFloat']['periodMs'],
 'face_lost_hold':lookup['layers.heart_crown.lifecycle.faceLost.holdMs']==layer['lifecycle']['faceLost']['holdMs'],
 'face_lost_fade':lookup['layers.heart_crown.lifecycle.faceLost.fadeMs']==layer['lifecycle']['faceLost']['fadeMs'],
 'rotation_x':lookup['layers.heart_crown.transform.rotationDeg.x']==layer['transform']['rotationDeg']['x'],
 'rotation_y':lookup['layers.heart_crown.transform.rotationDeg.y']==layer['transform']['rotationDeg']['y'],
 'rotation_z':lookup['layers.heart_crown.transform.rotationDeg.z']==layer['transform']['rotationDeg']['z'],
 'fit_ratio':lookup['layers.heart_crown.fit.ratio']==layer['fit']['ratio'],
 'head_center_x':lookup['layers.heart_crown.occlusion.headProxyEllipsoid.centerOffsetFW.x']==layer['occlusion']['headProxyEllipsoid']['centerOffsetFW']['x'],
 'head_center_y':lookup['layers.heart_crown.occlusion.headProxyEllipsoid.centerOffsetFW.y']==layer['occlusion']['headProxyEllipsoid']['centerOffsetFW']['y'],
 'head_center_z':lookup['layers.heart_crown.occlusion.headProxyEllipsoid.centerOffsetFW.z']==layer['occlusion']['headProxyEllipsoid']['centerOffsetFW']['z'],
 'metallic':lookup['layers.heart_crown.material.metallicFactor']==layer['material']['metallicFactor'],
 'roughness':lookup['layers.heart_crown.material.roughnessFactor']==layer['material']['roughnessFactor']
}
for idx,heart in enumerate(layer['geometryDesign']['heartLayout']):
    prefix=f'layers.heart_crown.geometryDesign.heartLayout.{idx}'
    checks_sync[f'heart_{idx+1:02d}_x']=lookup[prefix+'.centerFW.x']==heart['centerFW']['x']
    checks_sync[f'heart_{idx+1:02d}_y']=lookup[prefix+'.centerFW.y']==heart['centerFW']['y']
    checks_sync[f'heart_{idx+1:02d}_z']=lookup[prefix+'.centerFW.z']==heart['centerFW']['z']
    checks_sync[f'heart_{idx+1:02d}_width']=lookup[prefix+'.widthFW']==heart['widthFW']
    checks_sync[f'heart_{idx+1:02d}_rotation']=lookup[prefix+'.rotationDeg']==heart['rotationDeg']
assert all(checks_sync.values()), checks_sync
# Recompute all asset fingerprints after final model files exist.
asset_records=[]
for p in [ROOT/'thumbnail.png', atlas_path, *sorted((ROOT/'models').glob('*.glb'))]:
    data=p.read_bytes(); asset_records.append({'path':p.relative_to(ROOT).as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
manifest['assets']=asset_records
manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')

report={
 'package':'Heart Mood Filter',
 'validationStatus':'STATIC_VALIDATION_PASSED',
 'runtimeImportStatus':'NOT_SUPPORTED_BY_CURRENT_XELVIC_FILTER_PACKAGE_MANAGER',
 'physicalDeviceStatus':'NOT_TESTED',
 'heartCountPerTier':len(HEARTS),
 'lods':checks,
 'calibrationDefaultsSynchronized':checks_sync,
 'coordinateConvention':'+X subject-right; +Y up; +Z toward camera; no package mirroring',
 'occlusion':'Face mesh + head proxy ellipsoid enabled; ear/temple proxy disabled by scope',
 'renderStage':'OPAQUE_MODELS (Stage 5), after CLEAR_DEPTH_AND_OCCLUDERS (Stage 4)',
 'files':asset_records
}
(ROOT/'validation'/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

# Write package manifest/report one final time (manifest unchanged except hashes).
zip_path=Path('/mnt/data/Heart_Mood_Filter_RnD.zip')
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for p in sorted(ROOT.rglob('*')):
        if p.is_file(): z.write(p,p.relative_to(ROOT).as_posix())
print(json.dumps({'package_dir':str(ROOT),'zip':str(zip_path),'zip_bytes':zip_path.stat().st_size,'lod_metrics':metrics,'checks':checks,'calibration_sync':checks_sync,'files':[{'p':str(p.relative_to(ROOT)),'bytes':p.stat().st_size} for p in sorted(ROOT.rglob('*')) if p.is_file()]},indent=2))
