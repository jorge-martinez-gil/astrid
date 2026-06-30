import warnings; warnings.filterwarnings("ignore")
import sys, types, importlib.util, io, zipfile, numpy as np
from PIL import Image
st=types.ModuleType("streamlit"); st.__getattr__=lambda n:(lambda *a,**k:None); st.session_state={}
sys.modules["streamlit"]=st
for m in ["plotly","plotly.express","plotly.graph_objects","altair","matplotlib","matplotlib.pyplot"]:
    sys.modules.setdefault(m, types.ModuleType(m))
spec=importlib.util.spec_from_file_location("img_page","pages/03_Images.py")
M=importlib.util.module_from_spec(spec); sys.modules["img_page"]=M
try: spec.loader.exec_module(M)
except Exception: pass

def make_cfg():
    return M.AssessConfig(path_col=None, label_col=None, split_col=None, time_col=None,
        group_cols=[], id_cols=[], source_cols=[], condition_cols=[], annotator_cols=[],
        metadata={}, mode="Deep Scan", thresholds=M.PRESETS["Balanced (recommended)"],
        random_state=0, max_images=20000, sample_for_perceptual_dups=20000,
        sample_for_exif=200, max_pairs_for_near_dups=300000)

def imgs_to_zip(arrs, labels):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf:
        for i,(a,l) in enumerate(zip(arrs,labels)):
            im=Image.fromarray(a.astype("uint8"),mode="L").resize((64,64),Image.NEAREST)
            b=io.BytesIO(); im.save(b,format="PNG")
            zf.writestr(f"class{int(l)}/img_{i:05d}.png", b.getvalue())
    return buf.getvalue()

def image_health(arrs, labels):
    zb=imgs_to_zip(arrs,labels); cfg=make_cfg()
    img_df, ann, warn = M.read_zip_images(zb, cfg)
    meta = M.join_meta(img_df, None, cfg)
    rep = M.assess_all(img_df, meta, ann, cfg, zb)
    res = M.compute_metric_scores(rep, cfg)
    return res["total"], res["scores"], rep
