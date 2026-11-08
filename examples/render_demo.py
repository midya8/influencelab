"""Reproducible SVG built only from the full committed model audit."""
import argparse
import base64
import html
import json
from pathlib import Path
import sys
import zlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from influencelab.demo import demo


def render(report):
    a=report['undamped'];b=report['damped'];out=[]
    def text(x,y,s,size=15,color='#dce7fa',weight='normal',family='sans-serif'):
        out.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{color}" font-weight="{weight}">{html.escape(str(s))}</text>')
    def box(x,y,w,h):out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#1b293f" stroke="#32445d"/>')
    payload=base64.b64encode(zlib.compress(json.dumps(report,sort_keys=True,separators=(',',':'),allow_nan=False).encode())).decode()
    out.append('<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="900" viewBox="0 0 1120 900">')
    out.append(f'<metadata id="report-zlib-base64">{payload}</metadata><rect width="1120" height="900" fill="#0f172a"/>')
    text(28,40,'InfluenceLab | a derivative is not a deleted-data refit',27,weight='bold')
    text(28,68,'Convex binary logistic regression, 7 developed rows, original-N normalization, ridge 0.1.',15,'#9aaac1')
    box(26,88,1068,116)
    text(44,115,'One auditable objective',17,'#67e8f9','bold')
    text(44,143,'R(theta,w) = sum_i w_i loss_i(theta) / N + lambda ||theta||² / 2',17,family='monospace')
    text(44,171,'dtheta/dw_i = -H^-1 gradient(loss_i) / N; deletion uses delta w_i = -1.',15)
    text(44,190,'All coordinates, including the explicit intercept, are regularized. No hidden jitter.',13,'#9aaac1')
    box(26,220,1068,310)
    columns=[44,142,328,512,710,902]
    for j,s in enumerate(['row / label','x value','linear deletion','actual fixed-N','retained mean','damped linear']):text(columns[j],250,s,14,'#67e8f9','bold')
    for i,r in enumerate(a['rows']):
        y=284+31*i
        values=[f"{i} / {report['fixture']['y'][i]}",str(report['fixture']['X'][i][1]),f"{r['predicted_delete_query_loss_delta']:+.6f}",f"{r['actual_delete_query_loss_delta']:+.6f}",f"{r['retained_mean_query_loss_delta']:+.6f}",f"{b['rows'][i]['predicted_delete_query_loss_delta']:+.6f}"]
        for j,v in enumerate(values):text(columns[j],y,v,15,'#fbbf24' if i==5 else '#dce7fa',family='monospace')
    text(44,515,'All columns are changes in query loss. Actual refits are converged numerical solutions, not exact arithmetic.',13,'#9aaac1')
    box(26,546,518,142);box(558,546,536,142)
    row=a['rows'][5]
    text(44,575,'Visible finite-deletion error: row 5',17,'#67e8f9','bold')
    text(44,605,f"linear {row['predicted_delete_query_loss_delta']:+.6f} vs refit {row['actual_delete_query_loss_delta']:+.6f}",20,'#fbbf24',family='monospace')
    text(44,633,f"signed error {row['delete_query_loss_error']:+.6f}",17,family='monospace')
    text(44,662,'No claim that deleting a full row is infinitesimal.',13,'#9aaac1')
    text(576,575,'Normalization and damping are distinct',17,'#67e8f9','bold')
    text(576,605,'Retained-row mean with the same ridge is another objective.',14)
    text(576,630,'H + 0.5 I changes the inverse; the fitted theta stays fixed.',14)
    text(576,660,'It is not the derivative of a ridge-0.6 retrained model.',14,'#9aaac1')
    box(26,704,1068,133)
    text(44,733,'Convergence and solve witnesses',17,'#67e8f9','bold')
    fits=[a['fit']]+[r['fixed_normalizer_refit'] for r in a['rows']]+[r['retained_mean_refit'] for r in a['rows'] if 'retained_mean_refit' in r]
    text(44,764,f"max |gradient| across {len(fits)} fits: {max(f['max_abs_gradient'] for f in fits):.3e}",17,family='monospace')
    text(44,791,f"max influence solve residual: {max(r['max_abs_solve_residual'] for r in a['rows']):.3e}",17,family='monospace')
    text(44,818,'Full JSON preserves Hessians, Cholesky factors, parameter directions, refits and line-search histories.',13,'#9aaac1')
    text(28,865,'Standard influence functions with independent finite-difference and deletion references. Developed synthetic fixture.',14,'#9aaac1')
    text(28,886,'Local supplied-model sensitivity, not causal attribution, mislabeled-row proof, neural-network scaling or an unlearning guarantee.',13,'#9aaac1')
    out.append('</svg>')
    return '\n'.join(out)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    path=Path(__file__).resolve().parents[1]/'assets/influence-audit.svg';value=render(demo())
    if args.check:
        if path.read_text()!=value:raise SystemExit('SVG differs')
    else:path.write_text(value)
