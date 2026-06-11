#!/usr/bin/env python3
"""
Build the human-ranking evaluation bundle: images + data.json + index.html -> human_eval.zip
"""
import argparse
import json
import os
import random
import shutil
import zipfile
from pathlib import Path
from PIL import Image

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Image Editing Human Evaluation</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#1a1a2e;color:#e0e0e0;min-height:100vh}
.topbar{position:sticky;top:0;z-index:100;background:#16213e;padding:8px 16px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #0f3460;flex-wrap:wrap}
.progress-wrap{flex:1;min-width:200px;background:#0f3460;border-radius:8px;height:24px;position:relative;overflow:hidden}
.progress-fill{height:100%;background:linear-gradient(90deg,#e94560,#533483);transition:width .3s;border-radius:8px}
.progress-text{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:600}
.btn{padding:6px 14px;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600;transition:all .2s}
.btn-primary{background:#e94560;color:#fff}.btn-primary:hover{background:#c73e54}
.btn-secondary{background:#533483;color:#fff}.btn-secondary:hover{background:#6a4c9c}
.btn-outline{background:transparent;border:1px solid #e94560;color:#e94560}.btn-outline:hover{background:#e94560;color:#fff}
.caption-bar{background:#0f3460;padding:12px 20px;text-align:center;font-size:15px;line-height:1.6;border-bottom:1px solid #533483}
.caption-en{color:#e0e0e0;font-size:15px}
.caption-zh{color:#aaa;font-size:14px;margin-top:4px}
.caption-label{color:#e94560;font-weight:700;margin-right:8px}
.main{display:grid;grid-template-columns:1fr 3fr;gap:16px;padding:16px;max-width:1600px;margin:0 auto}
@media(max-width:900px){.main{grid-template-columns:1fr}}
.original-panel{text-align:center}
.original-panel h3{margin-bottom:8px;color:#aaa;font-size:13px}
.edits-panel{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.img-card{background:#16213e;border-radius:10px;padding:10px;text-align:center;border:2px solid transparent;transition:border-color .2s}
.img-card.ranked{border-color:#533483}
.img-card h4{margin-bottom:6px;color:#ccc;font-size:14px}
.img-card img{width:100%;border-radius:6px;cursor:pointer;transition:transform .2s}
.img-card img:hover{transform:scale(1.03)}
.rank-btns{display:flex;gap:6px;justify-content:center;margin-top:8px}
.rank-btn{width:48px;height:32px;border:2px solid #444;border-radius:6px;background:transparent;color:#ccc;cursor:pointer;font-weight:700;font-size:14px;transition:all .2s}
.rank-btn:hover{border-color:#e94560;color:#e94560}
.rank-btn.r1{border-color:#ffd700;color:#ffd700;background:rgba(255,215,0,.15)}
.rank-btn.r2{border-color:#c0c0c0;color:#c0c0c0;background:rgba(192,192,192,.15)}
.rank-btn.r3{border-color:#cd7f32;color:#cd7f32;background:rgba(205,127,50,.15)}
.nav{display:flex;justify-content:center;gap:12px;padding:16px;align-items:center}
.nav input{width:60px;text-align:center;background:#0f3460;border:1px solid #533483;color:#e0e0e0;padding:4px;border-radius:4px}
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:200;justify-content:center;align-items:center}
.modal-overlay.active{display:flex}
.modal{background:#16213e;border-radius:12px;padding:24px;max-width:700px;width:90%;max-height:85vh;overflow-y:auto}
.modal h2{margin-bottom:16px;color:#e94560}
table{width:100%;border-collapse:collapse;margin:12px 0}
th,td{padding:8px 12px;text-align:center;border:1px solid #333}
th{background:#0f3460;color:#e94560}
td{color:#e0e0e0}
.lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.9);z-index:300;justify-content:center;align-items:center;cursor:zoom-out}
.lightbox.active{display:flex}
.lightbox img{max-width:95vw;max-height:95vh;border-radius:8px}
.shortcuts{text-align:center;padding:8px;color:#666;font-size:11px}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700;margin-left:4px}
.badge-done{background:#2ecc71;color:#000}
.badge-skip{background:#e74c3c;color:#fff}
</style>
</head>
<body>
<div class="topbar">
  <div class="progress-wrap"><div class="progress-fill" id="progressFill"></div><div class="progress-text" id="progressText">0/0</div></div>
  <button class="btn btn-secondary" onclick="showStats()">Statistics</button>
  <button class="btn btn-primary" onclick="exportResults()">Export</button>
  <button class="btn btn-outline" onclick="resetConfirm()">Reset</button>
</div>
<div class="caption-bar"><div class="caption-en"><span class="caption-label">Caption:</span><span id="captionEn"></span></div><div class="caption-zh" id="captionZh"></div></div>
<div class="main">
  <div class="original-panel">
    <h3>Original (Reference)</h3>
    <img id="origImg" src="" style="width:100%;border-radius:8px;cursor:pointer" onclick="openLightbox(this.src)">
  </div>
  <div class="edits-panel">
    <div class="img-card" id="cardA"><h4>Image A</h4><img id="imgA" src="" onclick="openLightbox(this.src)"><div class="rank-btns"><button class="rank-btn" onclick="setRank('A',1)">1st</button><button class="rank-btn" onclick="setRank('A',2)">2nd</button><button class="rank-btn" onclick="setRank('A',3)">3rd</button></div></div>
    <div class="img-card" id="cardB"><h4>Image B</h4><img id="imgB" src="" onclick="openLightbox(this.src)"><div class="rank-btns"><button class="rank-btn" onclick="setRank('B',1)">1st</button><button class="rank-btn" onclick="setRank('B',2)">2nd</button><button class="rank-btn" onclick="setRank('B',3)">3rd</button></div></div>
    <div class="img-card" id="cardC"><h4>Image C</h4><img id="imgC" src="" onclick="openLightbox(this.src)"><div class="rank-btns"><button class="rank-btn" onclick="setRank('C',1)">1st</button><button class="rank-btn" onclick="setRank('C',2)">2nd</button><button class="rank-btn" onclick="setRank('C',3)">3rd</button></div></div>
  </div>
</div>
<div class="nav">
  <button class="btn btn-secondary" onclick="navigate(-1)">&lt; Prev (P)</button>
  <span>Sample <input id="jumpInput" type="number" min="1" onchange="jumpTo(this.value-1)"> / <span id="totalText">0</span></span>
  <button class="btn btn-primary" onclick="navigate(1)">Next (N) &gt;</button>
</div>
<div class="shortcuts">Shortcuts: 1/2/3 then A/B/C to rank | N=next P=prev | S=stats | E=export | Click image to zoom</div>
<div class="modal-overlay" id="statsModal" onclick="if(event.target===this)this.classList.remove('active')"><div class="modal" id="statsContent"></div></div>
<div class="lightbox" id="lightbox" onclick="this.classList.remove('active')"><img id="lightboxImg" src=""></div>

<script>
let data=null, state={currentIndex:0, rankings:{}, timestamps:{}};
const METHODS=["sdg","imdoc","fixed"];
let pendingKey=null;

async function init(){
  data=await(await fetch('./data.json')).json();
  const saved=localStorage.getItem('human_eval_'+data.seed);
  if(saved){try{state=JSON.parse(saved)}catch(e){}}
  document.getElementById('totalText').textContent=data.samples.length;
  renderSample(state.currentIndex);
}

function saveState(){
  localStorage.setItem('human_eval_'+data.seed, JSON.stringify(state));
}

function renderSample(idx){
  if(!data)return;
  idx=Math.max(0,Math.min(idx,data.samples.length-1));
  state.currentIndex=idx;
  const s=data.samples[idx];
  document.getElementById('captionEn').textContent=s.caption;
  document.getElementById('captionZh').textContent=s.caption_zh||'';
  document.getElementById('origImg').src=s.original;
  document.getElementById('jumpInput').value=idx+1;
  const order=s.display_order;
  const labels=['A','B','C'];
  labels.forEach((l,i)=>{
    const method=METHODS[order[i]];
    document.getElementById('img'+l).src=s.edits[method];
  });
  // restore ranks
  labels.forEach(l=>{
    const card=document.getElementById('card'+l);
    card.classList.remove('ranked');
    card.querySelectorAll('.rank-btn').forEach(b=>b.className='rank-btn');
  });
  const r=state.rankings[idx];
  if(r){
    Object.entries(r).forEach(([l,rank])=>{
      highlightRank(l,rank);
      document.getElementById('card'+l).classList.add('ranked');
    });
  }
  // timestamp
  if(!state.timestamps[idx])state.timestamps[idx]={start:Date.now()};
  updateProgress();
  saveState();
  // preload next
  if(idx+1<data.samples.length){
    const ns=data.samples[idx+1];
    [ns.original,...Object.values(ns.edits)].forEach(u=>{const i=new window.Image();i.src=u});
  }
}

function setRank(label,rank){
  const idx=state.currentIndex;
  if(!state.rankings[idx])state.rankings[idx]={};
  const r=state.rankings[idx];
  if(r[label]===rank){delete r[label];highlightRank(label,0);document.getElementById('card'+label).classList.remove('ranked')}
  else{r[label]=rank;highlightRank(label,rank);document.getElementById('card'+label).classList.add('ranked')}
  if(!state.timestamps[idx])state.timestamps[idx]={start:Date.now()};
  state.timestamps[idx].end=Date.now();
  saveState();
  updateProgress();
}

function highlightRank(label,rank){
  const btns=document.getElementById('card'+label).querySelectorAll('.rank-btn');
  btns.forEach(b=>b.className='rank-btn');
  if(rank>=1&&rank<=3)btns[rank-1].classList.add('r'+rank);
}

function navigate(d){
  renderSample(state.currentIndex+d);
}

function jumpTo(idx){renderSample(parseInt(idx)||0)}

function updateProgress(){
  const done=Object.keys(state.rankings).filter(k=>{const r=state.rankings[k];return r&&Object.keys(r).length===3}).length;
  const total=data.samples.length;
  const pct=Math.round(done/total*100);
  document.getElementById('progressFill').style.width=pct+'%';
  document.getElementById('progressText').textContent=done+'/'+total+' ('+pct+'%)';
}

function computeStats(){
  const completed=[];
  for(let i=0;i<data.samples.length;i++){
    const r=state.rankings[i];
    if(!r||Object.keys(r).length!==3)continue;
    const s=data.samples[i];
    const order=s.display_order;
    const labels=['A','B','C'];
    const methodRanks={};
    labels.forEach((l,li)=>{methodRanks[METHODS[order[li]]]=r[l]});
    completed.push(methodRanks);
  }
  const n=completed.length;
  if(!n)return null;
  const sums={},counts={},wins={};
  METHODS.forEach(m=>{sums[m]=0;counts[m]={1:0,2:0,3:0};wins[m]={}; METHODS.forEach(m2=>{wins[m][m2]=0})});
  completed.forEach(mr=>{
    METHODS.forEach(m=>{
      const rank=mr[m];
      sums[m]+=rank;
      counts[m][rank]=(counts[m][rank]||0)+1;
      METHODS.forEach(m2=>{
        if(m===m2)return;
        if(mr[m]<mr[m2])wins[m][m2]+=1;
        else if(mr[m]===mr[m2])wins[m][m2]+=0.5;
      });
    });
  });
  const stats={};
  METHODS.forEach(m=>{
    stats[m]={meanRank:(sums[m]/n).toFixed(3),r1:(counts[m][1]/n*100).toFixed(1),r2:(counts[m][2]/n*100).toFixed(1),r3:(counts[m][3]/n*100).toFixed(1)};
  });
  const winMatrix={};
  METHODS.forEach(m=>{winMatrix[m]={};METHODS.forEach(m2=>{winMatrix[m][m2]=m===m2?'-':(wins[m][m2]/n*100).toFixed(1)+'%'})});
  return{n,stats,winMatrix};
}

function showStats(){
  const r=computeStats();
  let html='<h2>Statistics</h2>';
  if(!r){html+='<p>No completed rankings yet.</p>';document.getElementById('statsContent').innerHTML=html;document.getElementById('statsModal').classList.add('active');return}
  html+='<p>Completed: '+r.n+'/'+data.samples.length+'</p>';
  html+='<h3 style="margin:16px 0 8px">Per-Method Rankings</h3>';
  html+='<table><tr><th>Method</th><th>Mean Rank</th><th>1st%</th><th>2nd%</th><th>3rd%</th></tr>';
  METHODS.forEach(m=>{const s=r.stats[m];html+='<tr><td><b>'+m.toUpperCase()+'</b></td><td>'+s.meanRank+'</td><td>'+s.r1+'%</td><td>'+s.r2+'%</td><td>'+s.r3+'%</td></tr>'});
  html+='</table>';
  html+='<h3 style="margin:16px 0 8px">Win Rate Matrix (row beats col)</h3>';
  html+='<table><tr><th></th>';METHODS.forEach(m=>{html+='<th>'+m.toUpperCase()+'</th>'});html+='</tr>';
  METHODS.forEach(m1=>{html+='<tr><td><b>'+m1.toUpperCase()+'</b></td>';METHODS.forEach(m2=>{html+='<td>'+r.winMatrix[m1][m2]+'</td>'});html+='</tr>'});
  html+='</table>';
  document.getElementById('statsContent').innerHTML=html;
  document.getElementById('statsModal').classList.add('active');
}

function exportResults(){
  const results=[];
  for(let i=0;i<data.samples.length;i++){
    const r=state.rankings[i];
    if(!r||Object.keys(r).length!==3)continue;
    const s=data.samples[i];
    const order=s.display_order;
    const labels=['A','B','C'];
    const labelToMethod={};
    labels.forEach((l,li)=>{labelToMethod[l]=METHODS[order[li]]});
    const methodRanking={};
    labels.forEach(l=>{methodRanking[labelToMethod[l]]=r[l]});
    const ts=state.timestamps[i]||{};
    results.push({sample_id:i,unique_name:s.unique_name,caption:s.caption,label_to_method:labelToMethod,rankings:r,method_ranking:methodRanking,duration_seconds:ts.end&&ts.start?Math.round((ts.end-ts.start)/1000):null});
  }
  const statsResult=computeStats();
  const exportData={export_time:new Date().toISOString(),num_completed:results.length,num_total:data.samples.length,results,summary:statsResult};
  const blob=new Blob([JSON.stringify(exportData,null,2)],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='human_eval_results_'+new Date().toISOString().slice(0,10)+'.json';
  a.click();
}

function resetConfirm(){
  if(confirm('Reset all rankings? This cannot be undone.')){
    state={currentIndex:0,rankings:{},timestamps:{}};
    localStorage.removeItem('human_eval_'+data.seed);
    renderSample(0);
  }
}

function openLightbox(src){
  document.getElementById('lightboxImg').src=src;
  document.getElementById('lightbox').classList.add('active');
}

document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  const k=e.key;
  if(k>='1'&&k<='3'){pendingKey=parseInt(k);return}
  if(pendingKey&&'aAbBcC'.includes(k)){setRank(k.toUpperCase(),pendingKey);pendingKey=null;return}
  if(k==='n'||k==='N'||k==='ArrowRight')navigate(1);
  else if(k==='p'||k==='P'||k==='ArrowLeft')navigate(-1);
  else if(k==='s'||k==='S')showStats();
  else if(k==='e'||k==='E')exportResults();
  else if(k==='Escape'){document.getElementById('statsModal').classList.remove('active');document.getElementById('lightbox').classList.remove('active')}
  pendingKey=null;
});

init();
</script>
</body>
</html>"""


def unique_name(filepath):
    parts = Path(filepath).parts
    if len(parts) >= 3:
        return f"{parts[-3]}_{parts[-2]}_{Path(filepath).stem}"
    elif len(parts) >= 2:
        return f"{parts[-2]}_{Path(filepath).stem}"
    return Path(filepath).stem


def main():
    parser = argparse.ArgumentParser(description="Prepare the human-ranking evaluation bundle")
    parser.add_argument("--sdg_results", default="outputs/v2_gpt_image_sdg_full/results_overlap5.jsonl")
    parser.add_argument("--imdoc_results", default="outputs/v2_gpt_image_imdoc_full/results_overlap5.jsonl")
    parser.add_argument("--fixed_results", default="outputs/v2_gpt_image_fixed_full/results_overlap5.jsonl")
    parser.add_argument("--output_dir", default="outputs/human_eval")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--image_quality", type=int, default=85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    # load three sources
    maps = {}
    for name, path in [("sdg", args.sdg_results), ("imdoc", args.imdoc_results), ("fixed", args.fixed_results)]:
        maps[name] = {}
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                maps[name][rec["filepath"]] = rec

    overlap = set(maps["sdg"]) & set(maps["imdoc"]) & set(maps["fixed"])
    samples = []
    for fp in sorted(overlap):
        edited = {m: maps[m][fp].get("edited_path", "") for m in ["sdg", "imdoc", "fixed"]}
        if all(os.path.exists(p) for p in [fp] + list(edited.values())):
            samples.append({"filepath": fp, "caption": maps["sdg"][fp].get("caption", ""), "edited": edited})

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"samples: {len(samples)}")

    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # copy images and generate data.json
    data_samples = []
    for i, s in enumerate(samples):
        uname = unique_name(s["filepath"])
        if (i + 1) % 100 == 0:
            print(f"  processing {i+1}/{len(samples)}...")

        # copy & resize 4 images
        for suffix, src in [("original", s["filepath"])] + [(m, s["edited"][m]) for m in ["sdg", "imdoc", "fixed"]]:
            dst = str(img_dir / f"{uname}_{suffix}.jpg")
            if not os.path.exists(dst):
                img = Image.open(src).convert("RGB")
                w, h = img.size
                if max(w, h) > args.image_size:
                    ratio = args.image_size / max(w, h)
                    img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
                img.save(dst, "JPEG", quality=args.image_quality)

        # randomize order
        order = [0, 1, 2]
        random.shuffle(order)

        data_samples.append({
            "id": i,
            "unique_name": uname,
            "caption": s["caption"],
            "original": f"images/{uname}_original.jpg",
            "edits": {
                "sdg": f"images/{uname}_sdg.jpg",
                "imdoc": f"images/{uname}_imdoc.jpg",
                "fixed": f"images/{uname}_fixed.jpg",
            },
            "display_order": order,
        })

    data_json = {
        "seed": args.seed,
        "num_samples": len(data_samples),
        "methods": ["sdg", "imdoc", "fixed"],
        "samples": data_samples,
    }

    with open(out_dir / "data.json", "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False)

    with open(out_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE)

    print(f"data.json: {len(data_samples)} samples")
    print(f"index.html: written")

    # package as zip
    zip_path = str(out_dir) + ".zip"
    print(f"packaging {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                arc_name = os.path.relpath(abs_path, out_dir.parent)
                zf.write(abs_path, arc_name)

    zip_size = os.path.getsize(zip_path) / 1024 / 1024
    print(f"\ndone!")
    print(f"  directory: {out_dir}")
    print(f"  ZIP: {zip_path} ({zip_size:.0f} MB)")
    print(f"  image: {len(data_samples) * 4} ")


if __name__ == "__main__":
    main()
