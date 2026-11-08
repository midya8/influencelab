import base64
import copy
import importlib.util
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import zlib
from influencelab.demo import demo

spec=importlib.util.spec_from_file_location('figure',Path('examples/render_demo.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class FigureTests(unittest.TestCase):
    def test_cells_positions_and_values(self):
        report=demo();root=ET.fromstring(module.render(report));cells={(int(t.attrib['x']),int(t.attrib['y'])):t.text for t in root.findall('{*}text')}
        for i,row in enumerate(report['undamped']['rows']):
            y=284+31*i
            expected=[f"{i} / {report['fixture']['y'][i]}",str(report['fixture']['X'][i][1]),f"{row['predicted_delete_query_loss_delta']:+.6f}",f"{row['actual_delete_query_loss_delta']:+.6f}",f"{row['retained_mean_query_loss_delta']:+.6f}",f"{report['damped']['rows'][i]['predicted_delete_query_loss_delta']:+.6f}"]
            for x,value in zip([44,142,328,512,710,902],expected):self.assertEqual(cells[x,y],value)
        self.assertEqual(cells[44,633],'signed error -0.102451')
        self.assertIn('15 fits',cells[44,764])

    def test_full_report_metadata_and_hidden_hessian_tamper(self):
        report=demo();root=ET.fromstring(module.render(report));payload=root.find('{*}metadata').text
        self.assertEqual(json.loads(zlib.decompress(base64.b64decode(payload))),report)
        other=copy.deepcopy(report);other['undamped']['rows'][0]['fixed_normalizer_refit']['hessian'][0][0]+=.1
        self.assertNotEqual(module.render(report),module.render(other))

    def test_byte_reproducibility(self):
        self.assertEqual(Path('assets/influence-audit.svg').read_text(),module.render(demo()))
