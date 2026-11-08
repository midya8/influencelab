import base64
import copy
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import zlib
from influencelab import path_demo

spec=importlib.util.spec_from_file_location('pathfigure',Path('examples/render_paths.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class PathFigureTests(unittest.TestCase):
    def test_table_convergence_cells_and_all_polyline_points(self):
        r=path_demo();root=ET.fromstring(module.render(r));cells={(float(t.attrib['x']),float(t.attrib['y'])):t.text for t in root.findall('{*}text')}
        for i,name in enumerate(['row5','mixed_group','all_rows']):
            a=r['paths'][name];expected=[name,f"{a['first_order_query_loss_delta']:+.6f}",f"{a['endpoint_query_loss_delta']:+.6f}",f"{a['integrated_query_loss_delta']:+.6f}",f"{a['integration_minus_endpoint_delta']:+.3e}"]
            for x,text in zip([44,230,424,618,824],expected):self.assertEqual(cells[x,154+36*i],text)
            for j,level in enumerate(a['levels']):self.assertEqual(cells[[230,490,780][i],735+25*j],f"{level['simpson_minus_endpoint_delta']:+.3e}")
        for poly in root.findall('{*}polyline'):
            nodes=r['paths'][poly.attrib['data-path']]['nodes'];points=[list(map(float,p.split(','))) for p in poly.attrib['points'].split()];self.assertEqual(len(points),17)
            for p,node in zip(points,nodes):
                self.assertAlmostEqual(p[0],102+932*node['t'],delta=.00051)
                self.assertAlmostEqual(p[1],580-(node['dquery_loss_dt']+.75)/2*220,delta=.00051)
        self.assertIn('sum of individual deletions -0.142898',cells[44,909])

    def test_full_hidden_path_hessian_metadata(self):
        r=path_demo();root=ET.fromstring(module.render(r));self.assertEqual(json.loads(zlib.decompress(base64.b64decode(root.find('{*}metadata').text))),r)
        other=copy.deepcopy(r);other['paths']['all_rows']['nodes'][7]['fit']['hessian'][0][1]+=.01
        self.assertNotEqual(module.render(other),module.render(r))

    def test_byte_regeneration(self):self.assertEqual(Path('assets/deletion-paths.svg').read_text(),module.render(path_demo()))
