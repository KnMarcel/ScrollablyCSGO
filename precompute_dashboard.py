"""
precompute_dashboard.py
───────────────────────
Run once locally: python precompute_dashboard.py

Generates processed/dashboard/<map>.json for each map.
Each JSON contains ALL pre-aggregated data the JS dashboard needs:
- All heatmap grids (every layer × side × nade combination)
- Weapon stats (radar axes, yield)
- Grenade KPIs per type per side
- Round-by-round momentum
- ATFF per map (cross-map comparison)
- Hitbox distribution
- Flash effectiveness

After running this, the Streamlit app serves each JSON once per map
and the entire dashboard runs in JS with zero Python roundtrips.
"""

PROCESSED_DIR = r"C:\Users\marce\Desktop\CSGO_Analytics\processed"
DASHBOARD_DIR = r"C:\Users\marce\Desktop\ScrollablyCSGO\dashboard"

import duckdb
import pandas as pd
import numpy as np
import os
import json
import time
from scipy.ndimage import gaussian_filter

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
HEATMAP_DIR   = os.path.join(PROCESSED_DIR, "heatmaps")
DASHBOARD_DIR = os.path.join(PROCESSED_DIR, "dashboard")
MAP_DATA_CSV  = os.path.join(BASE_DIR, "map_data.csv")

BINS       = 100   # lower res for JSON (smaller files, still looks great)
ATFF_CAP   = 115
GUN_TYPES  = {"Rifle","SMG","Pistol","Heavy","Sniper Rifle"}
NADE_MERGE = {"Molotov":"Fire","Incendiary":"Fire"}
SIDES      = {"All":None,"CT":"CounterTerrorist","T":"Terrorist"}
WEAPONS    = ["AK-47","M4A4","M4A1","AWP","Glock","USP","Deagle",
               "Desert Eagle","MP7","MP9","P250","SG553","AUG","Famas","Galil"]

os.makedirs(DASHBOARD_DIR, exist_ok=True)
con = duckdb.connect()

def load_parquet(f):
    p = os.path.join(PROCESSED_DIR, f)
    if not os.path.exists(p):
        print(f"  ⚠ Not found: {p}")
        return pd.DataFrame()
    return con.execute(f"SELECT * FROM read_parquet('{p}')").df()

print("Loading map data...")
map_data_path = os.path.join(PROCESSED_DIR, "map_data.parquet")
if os.path.exists(map_data_path):
    map_data = con.execute(f"SELECT * FROM read_parquet('{map_data_path}')").df()
    if "column0" in map_data.columns:
        map_data = map_data.rename(columns={"column0":"map"})
else:
    map_data = pd.read_csv(MAP_DATA_CSV, index_col=0)
    map_data.index.name = "map"
    map_data = map_data.reset_index()

print("Loading raw data...")
t0 = time.time()
meta     = load_parquet("meta.parquet")
dmg_all  = load_parquet("dmg.parquet")
gren_all = load_parquet("grenades.parquet")
kill_all = load_parquet("kills.parquet")

# Add map column
if "map" not in dmg_all.columns and "file" in dmg_all.columns:
    fm = meta[["file","map"]].drop_duplicates()
    dmg_all  = dmg_all.merge(fm, on="file", how="left")
    gren_all = gren_all.merge(fm, on="file", how="left")
    kill_all = kill_all.merge(fm, on="file", how="left")

# Merge fire nades
if "nade" in gren_all.columns:
    gren_all["nade"] = gren_all["nade"].replace(NADE_MERGE)

print(f"Loaded in {time.time()-t0:.1f}s")

# ATFF for all maps (for the cross-map strip)
print("Computing ATFF for all maps...")
atff_all = {}
for _, mr in map_data.iterrows():
    mn = mr["map"]
    d = dmg_all[dmg_all["map"]==mn] if "map" in dmg_all.columns else dmg_all
    if d.empty or "hp_dmg" not in d.columns:
        continue
    first = d[d["hp_dmg"]>0].groupby(["file","round"])["seconds"].min()
    first = first[first < ATFF_CAP]
    if len(first)>0:
        atff_all[mn] = round(float(first.mean()),2)

print(f"  ATFF: {atff_all}")


def world_to_bin(sx,sy,ex,ey,xs,ys,bins=BINS):
    nx = ((xs-sx)/(ex-sx)).clip(0,1)
    ny = ((ys-sy)/(ey-sy)).clip(0,1)
    col = ((1-ny)*(bins-1)).clip(0,bins-1).astype(int)
    row = (nx*(bins-1)).clip(0,bins-1).astype(int)
    return col, row

def make_grid(df, xcol, ycol, mr, bins=BINS):
    sx,sy = float(mr["StartX"]),float(mr["StartY"])
    ex,ey = float(mr["EndX"]),  float(mr["EndY"])
    v = df[[xcol,ycol]].dropna()
    v = v[(v[xcol]!=0)|(v[ycol]!=0)]
    if v.empty:
        return None
    col,row = world_to_bin(sx,sy,ex,ey,v[xcol].values,v[ycol].values,bins)
    g = np.zeros((bins,bins),dtype=np.float32)
    np.add.at(g,(row,col),1)
    g = gaussian_filter(g,sigma=2.0)
    mx = g.max()
    if mx>0: g = g/mx
    # Sparse: only store values > threshold
    mask = g > 0.02
    rows,cols = np.where(mask)
    vals = (g[rows,cols]*255).astype(np.uint8)
    return {"r":rows.tolist(),"c":cols.tolist(),"v":vals.tolist(),"bins":bins}

def weapon_stats(D, K, M, map_name):
    """Returns radar + yield data for top guns."""
    if D.empty or "wp" not in D.columns:
        return []
    D_guns = D[D["wp_type"].isin(GUN_TYPES)] if "wp_type" in D.columns else D
    top = D_guns["wp"].value_counts().head(6).index.tolist()
    rows = []
    n_rounds = max(len(M),1)
    for wp in top:
        sub  = D_guns[D_guns["wp"]==wp]
        hits = len(sub)
        if hits==0: continue
        hs   = (sub["hitbox"]=="Head").sum()/hits*100 if "hitbox" in sub.columns else 0
        avgd = float(sub["hp_dmg"].mean()) if "hp_dmg" in sub.columns else 0
        pen  = 0
        if "arm_dmg" in sub.columns:
            td = sub["hp_dmg"]+sub["arm_dmg"]
            pen = float((sub["arm_dmg"]/td.replace(0,np.nan)).mean()*100)
        wk = len(K[K["wp"]==wp]) if "wp" in K.columns else 0
        leth = (1/(hits/wk))*100 if wk>0 else 0
        rng = 0
        if all(c in sub.columns for c in ["att_pos_x","att_pos_y","vic_pos_x","vic_pos_y"]):
            c = sub[["att_pos_x","att_pos_y","vic_pos_x","vic_pos_y"]].dropna()
            if len(c)>0:
                rng = float(np.sqrt((c["att_pos_x"]-c["vic_pos_x"])**2+(c["att_pos_y"]-c["vic_pos_y"])**2).mean())
        vol = hits/n_rounds
        rows.append({"wp":wp,"hs":round(hs,1),"dmg":round(avgd,1),
                     "pen":round(pen,1),"leth":round(leth,1),
                     "range":round(rng,1),"vol":round(vol,2),"hits":hits})
    return rows

def hitbox_dist(D):
    if D.empty or "hitbox" not in D.columns: return {}
    vc = D["hitbox"].value_counts()
    tot = len(D)
    return {k:round(v/tot*100,1) for k,v in vc.items()} if tot>0 else {}

def momentum_data(D):
    if D.empty or "round" not in D.columns or "att_side" not in D.columns:
        return []
    rd = D.groupby(["round","att_side"])["hp_dmg"].sum().unstack(fill_value=0).reset_index()
    ct = "CounterTerrorist"; t = "Terrorist"
    if ct not in rd.columns: rd[ct]=0
    if t  not in rd.columns: rd[t]=0
    rd["delta"] = (rd[ct]-rd[t]).rolling(5,min_periods=1).mean().round(0)
    return [{"round":int(r),"delta":float(d)} for r,d in zip(rd["round"],rd["delta"])]

def flash_eff(G, K, M):
    if G.empty or K.empty or "seconds" not in G.columns or "seconds" not in K.columns:
        return 0.0
    if "start_seconds" not in M.columns: return 0.0
    rs = M[["file","round","start_seconds"]].drop_duplicates()
    fl = G[G["nade"]=="Flash"][["file","round","seconds"]].copy()
    fl = fl.merge(rs,on=["file","round"],how="left")
    fl["fs"] = fl["seconds"]-fl["start_seconds"]
    fl = fl[["file","round","fs"]].dropna()
    if len(fl)==0: return 0.0
    Ki = K[["file","round","seconds"]].copy()
    Ki["kid"] = range(len(Ki))
    mg = Ki.merge(fl,on=["file","round"],how="inner")
    mg["dt"] = mg["seconds"]-mg["fs"]
    v = mg[(mg["dt"]>=0)&(mg["dt"]<=3)]
    return round(v["kid"].nunique()/len(fl)*100,1)

def nade_kpis(G):
    """Per-type KPIs split by side."""
    if G.empty or "nade" not in G.columns: return {}
    out = {}
    for nade in G["nade"].unique():
        sub = G[G["nade"]==nade]
        out[nade] = {
            "total": len(sub),
            "dmg_total": int(sub["hp_dmg"].sum()) if "hp_dmg" in sub.columns else 0,
            "dmg_avg":   round(float(sub["hp_dmg"].mean()),1) if "hp_dmg" in sub.columns and len(sub)>0 else 0,
            "ct": len(sub[sub["att_side"]=="CounterTerrorist"]) if "att_side" in sub.columns else 0,
            "t":  len(sub[sub["att_side"]=="Terrorist"]) if "att_side" in sub.columns else 0,
        }
    return out


# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
for _, mr in map_data.iterrows():
    mn = mr["map"]
    out_path = os.path.join(DASHBOARD_DIR, f"{mn}.json")

    t1 = time.time()
    print(f"\n⚙  {mn}...", flush=True)

    D = dmg_all[dmg_all["map"]==mn].copy()  if "map" in dmg_all.columns  else dmg_all.copy()
    G = gren_all[gren_all["map"]==mn].copy() if "map" in gren_all.columns else gren_all.copy()
    K = kill_all[kill_all["map"]==mn].copy() if "map" in kill_all.columns else kill_all.copy()
    M = meta[meta["map"]==mn].copy()         if "map" in meta.columns     else meta.copy()

    payload = {
        "map": mn,
        "meta": {
            "rx": float(mr["ResX"]), "ry": float(mr["ResY"]),
            "n_rounds": len(M), "n_kills": len(K), "n_hits": len(D),
        },
        "atff_all": atff_all,
        "atff_map": atff_all.get(mn,0),
        "heatmaps": {},
        "weapons": weapon_stats(D,K,M,mn),
        "hitbox":  hitbox_dist(D),
        "momentum": momentum_data(D),
        "flash_eff": flash_eff(G,K,M),
        "nade_kpis": nade_kpis(G),
    }

    # ── HEATMAP GRIDS ─────────────────────────────────────────────────────────
    hm = payload["heatmaps"]

    # Damage — all sides
    hm["dmg_vic"]    = make_grid(D,"vic_pos_x","vic_pos_y",mr)
    hm["dmg_att"]    = make_grid(D,"att_pos_x","att_pos_y",mr)

    # Damage — per side
    for side_key, side_val in [("ct","CounterTerrorist"),("t","Terrorist")]:
        if "att_side" in D.columns:
            sub = D[D["att_side"]==side_val]
            hm[f"dmg_vic_{side_key}"] = make_grid(sub,"vic_pos_x","vic_pos_y",mr)
            hm[f"dmg_att_{side_key}"] = make_grid(sub,"att_pos_x","att_pos_y",mr)

    # Grenades — landed + thrown, all and per type
    if not G.empty and "nade_land_x" in G.columns:
        hm["nade_land_all"]    = make_grid(G,"nade_land_x","nade_land_y",mr)
        hm["nade_thrown_all"]  = make_grid(G,"att_pos_x","att_pos_y",mr)
        for nade_type in G["nade"].dropna().unique():
            sub = G[G["nade"]==nade_type]
            key = nade_type.replace(" ","_")
            hm[f"nade_land_{key}"]   = make_grid(sub,"nade_land_x","nade_land_y",mr)
            hm[f"nade_thrown_{key}"] = make_grid(sub,"att_pos_x","att_pos_y",mr)
            # Per side
            for side_key,side_val in [("ct","CounterTerrorist"),("t","Terrorist")]:
                if "att_side" in sub.columns:
                    ss = sub[sub["att_side"]==side_val]
                    hm[f"nade_land_{key}_{side_key}"]   = make_grid(ss,"nade_land_x","nade_land_y",mr)
                    hm[f"nade_thrown_{key}_{side_key}"] = make_grid(ss,"att_pos_x","att_pos_y",mr)

    # Weapon kill positions (top 6)
    if not K.empty and all(c in K.columns for c in ["wp","att_pos_x","att_pos_y"]):
        top_wp = D["wp"].value_counts().head(6).index.tolist() if "wp" in D.columns else []
        for wp in top_wp:
            ks = K[K["wp"]==wp]
            key = wp.replace("-","").replace(" ","_").replace("/","")
            hm[f"kills_{key}"] = make_grid(ks,"att_pos_x","att_pos_y",mr)

    # Remove None grids
    hm_clean = {k:v for k,v in hm.items() if v is not None}
    payload["heatmaps"] = hm_clean

    # Serialize
    with open(out_path,"w") as f:
        json.dump(payload,f,separators=(",",":"))

    size_kb = os.path.getsize(out_path)/1024
    print(f"  ✓ {time.time()-t1:.1f}s → {size_kb:.0f} KB ({len(hm_clean)} heatmap layers)")

print(f"\n✅ All done. Files in: {DASHBOARD_DIR}")
