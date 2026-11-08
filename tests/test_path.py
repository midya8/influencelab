import copy
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from influencelab import audit, fit, trace_deletion, path_demo
from influencelab.core import dot, point
from influencelab.path import _simpson
from test_core import X,Y


class PathTests(unittest.TestCase):
    def test_quadrature_against_independent_polynomial_integrals(self):
        for degree in range(4):
            for n in [2,4,8,16,64]:
                self.assertAlmostEqual(_simpson([(i/n)**degree for i in range(n+1)]),1/(degree+1),delta=2e-16)
        self.assertGreater(_simpson([(i/2)**4 for i in range(3)]),.2)

    def test_row5_integrated_vs_first_order_and_full_levels(self):
        r=trace_deletion(X,Y,[1,2],1,[5]);levels=r['levels']
        self.assertEqual([v['segments'] for v in levels],[2,4,8,16])
        self.assertAlmostEqual(r['first_order_query_loss_delta'],.20320426327204985)
        self.assertAlmostEqual(r['endpoint_query_loss_delta'],.30565521120251515)
        self.assertAlmostEqual(r['integrated_query_loss_delta'],.3056553492591347)
        self.assertLess(abs(r['integration_minus_endpoint_delta']),1.4e-7)
        for a,b in zip(levels,levels[1:]):
            self.assertAlmostEqual(b['simpson_minus_previous'],b['simpson']-a['simpson'])
            # Fixture-specific, not a guaranteed monotone quadrature theorem.
            self.assertLess(abs(b['simpson_minus_endpoint_delta']),abs(a['simpson_minus_endpoint_delta']))

    def test_all_rows_sign_reversal_and_unhidden_quadrature_error(self):
        r=trace_deletion(X,Y,[1,2],1,list(range(len(X))))
        self.assertLess(r['first_order_query_loss_delta'],0)
        self.assertGreater(r['endpoint_query_loss_delta'],0)
        self.assertEqual(r['nodes'][-1]['fit']['theta'],[0.,0.])
        self.assertAlmostEqual(r['endpoint_query_loss_delta'],math.log(2)-r['nodes'][0]['query_loss'])
        self.assertGreater(abs(r['integration_minus_endpoint_delta']),.0007)
        self.assertAlmostEqual(r['nodes'][-1]['dquery_loss_dt'],15/14)

    def test_group_nonadditivity_versus_individual_actual_refits(self):
        r=path_demo()['paths']['mixed_group']
        self.assertAlmostEqual(r['endpoint_query_loss_delta'],-.35750696527574705)
        self.assertAlmostEqual(r['sum_of_individual_endpoint_deltas'],.30565521120251515-.44855337973180787)
        self.assertGreater(abs(r['group_minus_sum_of_individual_deltas']),.2)
        self.assertAlmostEqual(r['group_minus_sum_of_individual_deltas'],r['endpoint_query_loss_delta']-r['sum_of_individual_endpoint_deltas'])

    def test_all_nodes_full_weights_convergence_hessian_and_residual(self):
        r=trace_deletion(X,Y,[1,2],1,[5,6])
        for node in r['nodes']:
            self.assertEqual(node['fit']['weights'],[1-node['t'] if i in [5,6] else 1 for i in range(len(X))])
            self.assertEqual(node['fit']['normalizer'],len(X));self.assertEqual(node['fit']['ridge'],.1)
            self.assertLessEqual(node['fit']['max_abs_gradient'],1e-10)
            self.assertLess(node['max_abs_solve_residual'],1e-15)
            for j in range(2):
                self.assertAlmostEqual(dot(node['fit']['hessian'][j],node['dtheta_dt']),node['rhs'][j])
                independent=sum(point(node['fit']['theta'],X[i],Y[i])[1][j] for i in [5,6])/len(X)
                self.assertAlmostEqual(independent,node['rhs'][j])
            self.assertAlmostEqual(dot(node['query_gradient'],node['dtheta_dt']),node['dquery_loss_dt'])

    def test_interior_parameter_and_query_derivatives_from_retraining(self):
        r=trace_deletion(X,Y,[1,2],1,[5,6])
        for node in r['nodes'][4:13:4]:
            eps=1e-5;t=node['t'];wp=[1-(t+eps) if i in [5,6] else 1 for i in range(len(X))];wm=[1-(t-eps) if i in [5,6] else 1 for i in range(len(X))]
            a=fit(X,Y,weights=wp,tolerance=1e-12)['theta'];b=fit(X,Y,weights=wm,tolerance=1e-12)['theta']
            for estimate,value in zip([(v-w)/(2*eps) for v,w in zip(a,b)],node['dtheta_dt']):self.assertAlmostEqual(estimate,value,delta=2e-8)
            estimate=(point(a,[1,2],1)[0]-point(b,[1,2],1)[0])/(2*eps)
            self.assertAlmostEqual(estimate,node['dquery_loss_dt'],delta=2e-8)

    def test_initial_direction_equals_sum_of_individual_deletion_predictions(self):
        base=audit(X,Y,[1,2],1,refit=False)
        for group in [[5],[5,6],list(range(7))]:
            r=trace_deletion(X,Y,[1,2],1,group,segments=2)
            for j in range(2):self.assertAlmostEqual(r['nodes'][0]['dtheta_dt'][j],sum(base['rows'][i]['predicted_delete_theta_delta'][j] for i in group))
            self.assertAlmostEqual(r['first_order_query_loss_delta'],sum(base['rows'][i]['predicted_delete_query_loss_delta'] for i in group))

    def test_endpoint_matches_direct_group_refit_same_normalization(self):
        r=trace_deletion(X,Y,[1,2],1,[2,4,5],segments=4)
        direct=fit(X,Y,weights=[0 if i in [2,4,5] else 1 for i in range(7)])
        self.assertEqual(r['nodes'][-1]['fit'],direct)
        self.assertAlmostEqual(r['endpoint_query_loss_delta'],point(direct['theta'],[1,2],1)[0]-r['nodes'][0]['query_loss'])

    def test_group_order_and_input_nonmutation_rng(self):
        x,y,group=copy.deepcopy(X),Y[:],[6,5];state=random.getstate()
        a=trace_deletion(x,y,[1,2],1,group);b=trace_deletion(X,Y,[1,2],1,[5,6])
        self.assertEqual(a,b);self.assertEqual(x,X);self.assertEqual(y,Y);self.assertEqual(group,[6,5]);self.assertEqual(state,random.getstate())

    def test_permutation_and_orthogonal_chart(self):
        a=trace_deletion(X,Y,[1,2],1,[5,6],segments=4);order=[4,1,6,0,2,5,3]
        b=trace_deletion([[X[i][1],-X[i][0]] for i in order],[Y[i] for i in order],[2,-1],1,[order.index(5),order.index(6)],segments=4)
        for p,q in zip(a['nodes'],b['nodes']):self.assertAlmostEqual(p['dquery_loss_dt'],q['dquery_loss_dt'])
        self.assertAlmostEqual(a['integrated_query_loss_delta'],b['integrated_query_loss_delta'])

    def test_single_row_all_deleted_and_zero_features(self):
        for x,y,q,label in [([[1]],[1],[1],1),([[0],[0]],[0,1],[0],0)]:
            r=trace_deletion(x,y,q,label,list(range(len(x))),segments=2)
            self.assertEqual(r['nodes'][-1]['fit']['theta'],[0.])
            if q==[0]:self.assertEqual(r['integrated_query_loss_delta'],0);self.assertEqual(r['endpoint_query_loss_delta'],0)

    def test_finest_grid_fits_reused_at_all_quadrature_levels(self):
        with patch('influencelab.path.fit',wraps=fit) as wrapper:
            r=trace_deletion(X,Y,[1,2],1,[5],segments=64)
            self.assertEqual(wrapper.call_count,65)
        self.assertEqual([v['segments'] for v in r['levels']],[2,4,8,16,32,64])
        self.assertEqual(len(r['nodes']),65)
        self.assertLess(abs(r['integration_minus_endpoint_delta']),1e-9)

    def test_validation(self):
        for group in [[],[0,0],[True],[-1],[7],[0.0],'5']:
            with self.subTest(group=group),self.assertRaises(ValueError):trace_deletion(X,Y,[1,2],1,group)
        for segments in [True,0,1,3,6,128,4.0]:
            with self.subTest(segments=segments),self.assertRaises(ValueError):trace_deletion(X,Y,[1,2],1,[5],segments=segments)
        with self.assertRaises(ValueError):trace_deletion(X,Y,[1],1,[5])
        with self.assertRaises(ValueError):trace_deletion(X,Y,[1,2],True,[5])

    def test_cli_demo_input_and_strict_schema(self):
        committed=json.loads(Path('examples/path-demo.json').read_text());self.assertEqual(path_demo(),committed)
        p=subprocess.run([sys.executable,'-m','influencelab','path-demo'],capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),committed)
        source=json.loads(Path('examples/path-input.json').read_text())
        p=subprocess.run([sys.executable,'-m','influencelab','trace','examples/path-input.json'],capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
        expected=trace_deletion(X,Y,[1,2],1,[5]);self.assertEqual(json.loads(p.stdout),expected)
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'input.json'
            for request in [dict(source,indices=[5,5]),dict(source,segments=3),dict(source,version=True),dict(source,refit=True)]:
                f.write_text(json.dumps(request));p=subprocess.run([sys.executable,'-m','influencelab','trace',str(f)],capture_output=True,text=True);self.assertEqual(p.returncode,2);self.assertFalse(p.stdout)

if __name__=='__main__':unittest.main()
