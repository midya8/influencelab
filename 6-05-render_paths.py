"""Reproduce the complete finite-deletion path audit figure."""
import argparse
import base64
import html
import json
from pathlib import Path
import sys
import zlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from influencelab.path import path_demo


def render(report):
    paths=report['paths'];out=[]
    def text(x,y,s,size=15,color='#dce7fa',weight='normal',family='sans-serif'):
        out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" font-family="{family}">{html.escape(str(s))}</text>')
    def box(x,y,w,h):out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#1b293f" stroke="#32445d"/>')
    payload=base64.b64encode(zlib.compress(json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode())).decode()
    out.append('<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="1040" viewBox="0 0 1120 1040">')
    out.append(f'<metadata id="report-zlib-base64">{payload}</metadata><rect width="1120" height="1040" fill="#0f172a"/>')
    text(28,40,'InfluenceLab | follow the deletion, do not freeze the derivative',25,weight='bold')
    text(28,69,'Selected sample weights follow 1-t. At each node: a converged refit and a new implicit derivative.',15,'#9aaac1')
    box(26,88,1068,185)
    columns=[44,230,424,618,824]
    for x,s in zip(columns,['selected rows','first-order delta','endpoint delta','Simpson16 delta','integral - endpoint']):text(x,118,s,14,'#67e8f9','bold')
    colors={'row5':'#67e8f9','mixed_group':'#c4b5fd','all_rows':'#fbbf24'}
    for i,name in enumerate(['row5','mixed_group','all_rows']):
        r=paths[name];values=[name,f"{r['first_order_query_loss_delta']:+.6f}",f"{r['endpoint_query_loss_delta']:+.6f}",f"{r['integrated_query_loss_delta']:+.6f}",f"{r['integration_minus_endpoint_delta']:+.3e}"]
        for x,value in zip(columns,values):text(x,154+36*i,value,16,colors[name],family='monospace')
    text(44,253,'Numerical endpoint refits, not symbolic arithmetic. All changes concern the same supplied query loss.',13,'#9aaac1')
    box(26,290,1068,342)
    text(44,318,'The query-loss derivative changes along the path',18,'#67e8f9','bold')
    for i,name in enumerate(['row5','mixed_group','all_rows']):text(530+180*i,318,name,14,colors[name],'bold')
    def xy(t,value):return 102+932*t,580-(value+.75)/2*220
    for value in [-.5,0,.5,1]:
        _,y=xy(0,value);out.append(f'<line x1="102" y1="{y:.3f}" x2="1034" y2="{y:.3f}" stroke="#40516a"/>');text(44,y+5,f'{value:+.1f}',13,'#9aaac1')
    for t in [0,.25,.5,.75,1]:
        x,_=xy(t,0);text(x-8,604,str(t),13,'#9aaac1')
    for name in ['row5','mixed_group','all_rows']:
        points=' '.join(f'{xy(n["t"],n["dquery_loss_dt"])[0]:.3f},{xy(n["t"],n["dquery_loss_dt"])[1]:.3f}' for n in paths[name]['nodes'])
        out.append(f'<polyline data-path="{name}" points="{points}" fill="none" stroke="{colors[name]}" stroke-width="3"/>')
    text(925,625,'deletion t: 0 to 1',12,'#9aaac1')
    box(26,648,1068,185)
    text(44,677,'Quadrature gaps remain visible: Simpson estimate minus actual endpoint loss change',17,'#67e8f9','bold')
    for x,s in zip([44,230,490,780],['segments','row5','mixed group [5,6]','all rows']):text(x,709,s,14,'#9aaac1','bold')
    for i,n in enumerate([2,4,8,16]):
        text(44,735+25*i,n,14,family='monospace')
        for x,name in zip([230,490,780],['row5','mixed_group','all_rows']):text(x,735+25*i,f"{paths[name]['levels'][i]['simpson_minus_endpoint_delta']:+.3e}",14,colors[name],family='monospace')
    box(26,850,1068,127)
    r=paths['mixed_group']
    text(44,879,'Group deletion is not a sum of separately refitted deletions',18,'#67e8f9','bold')
    text(44,909,f"delete [5,6] {r['endpoint_query_loss_delta']:+.6f} vs sum of individual deletions {r['sum_of_individual_endpoint_deltas']:+.6f}",17,family='monospace')
    text(44,939,'Deleting all rows even reverses the sign of the frozen first-order prediction on this developed fixture.',14)
    text(44,960,'51 path fits, 3 groups x 17 nodes. Coarse quadrature levels reuse the fine-grid fits.',13,'#9aaac1')
    text(28,1004,'Fixed original-N normalization and ridge0.1. No damping, warm starts, causal claim or numerical error certification.',14,'#9aaac1')
    text(28,1026,'Numerical path sensitivity is not machine unlearning, a scalable deletion algorithm or a guarantee about other datasets.',13,'#9aaac1')
    out.append('</svg>');return '\n'.join(out)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();path=Path(__file__).resolve().parents[1]/'assets/deletion-paths.svg';value=render(path_demo())
    if args.check:
        if path.read_text()!=value:raise SystemExit('path SVG differs')
    else:path.write_text(value)
