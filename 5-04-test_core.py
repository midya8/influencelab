import copy
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from influencelab import audit, fit
from influencelab.core import _objective, dot, point, solve
from influencelab.demo import demo

X = [[1, -2], [1, -1], [1, -.5], [1, .5], [1, 1], [1, 2], [1, 3]]
Y = [0, 0, 1, 0, 1, 1, 0]


class CoreTests(unittest.TestCase):
    def close_vectors(self, a, b, tolerance=1e-7):
        self.assertEqual(len(a), len(b))
        for v, w in zip(a, b): self.assertAlmostEqual(v, w, delta=tolerance)

    def test_point_against_independent_log_probability(self):
        for z in [-20, -3, 0, 2, 20]:
            for label in [0, 1]:
                loss, gradient, curvature, p = point([z], [1], label)
                independent = 1 / (1 + math.exp(-z))
                self.assertAlmostEqual(p, independent)
                self.assertAlmostEqual(loss, -math.log(independent if label else 1-independent), delta=4e-8)
                self.assertAlmostEqual(gradient[0], independent-label)
                self.assertAlmostEqual(curvature, independent*(1-independent))

    def test_point_extreme_logits(self):
        for z in [-1000, 1000]:
            for label in [0, 1]:
                loss, gradient, curvature, p = point([z], [1], label)
                self.assertTrue(all(math.isfinite(v) for v in [loss, *gradient, curvature, p]))
                self.assertEqual(loss, max((1-2*label)*z, 0))
        self.assertNotEqual(point([40], [1], 1)[1][0], 0)
        self.assertGreater(point([40], [1], 1)[2], 0)

    def test_objective_all_gradient_and_hessian_entries_finite_difference(self):
        theta = [.4, -.7, .2]
        x = [[1, -2, 3], [1, 1, -.5], [1, 0, 2]]
        y, weights = [0, 1, 1], [.2, 1.3, 0]
        value, gradient, hessian = _objective(theta, x, y, weights, .17, 5)
        eps = 1e-5
        for j in range(3):
            a, b = theta[:], theta[:]
            a[j] += eps; b[j] -= eps
            plus = _objective(a, x, y, weights, .17, 5)
            minus = _objective(b, x, y, weights, .17, 5)
            self.assertAlmostEqual((plus[0]-minus[0])/(2*eps), gradient[j], delta=1e-10)
            for k in range(3):
                self.assertAlmostEqual((plus[1][k]-minus[1][k])/(2*eps), hessian[k][j], delta=1e-10)

    def test_fit_converges_history_and_strong_convex_bound(self):
        result = fit(X, Y)
        self.assertTrue(result['converged'])
        self.assertLessEqual(result['max_abs_gradient'], result['tolerance'])
        history = result['history']
        for a,b in zip(history,history[1:]): self.assertLessEqual(b['objective'],a['objective']+2e-15)
        tight = fit(X, Y, tolerance=1e-12)
        distance = math.dist(result['theta'], tight['theta'])
        self.assertLessEqual(distance, result['parameter_error_l2_bound']+tight['parameter_error_l2_bound']+1e-15)

    def test_cholesky_reconstruction_and_independent_gaussian_elimination(self):
        r = audit(X, Y, [1,2], 1, refit=False)
        h, l = r['effective_hessian'], r['effective_cholesky']
        for j in range(2):
            for k in range(2): self.assertAlmostEqual(dot(l[j],l[k]),h[j][k])
        for row in r['rows']:
            rhs = [-v/len(X) for v in row['sample_gradient']]
            determinant = h[0][0]*h[1][1]-h[0][1]*h[1][0]
            direct = [(rhs[0]*h[1][1]-h[0][1]*rhs[1])/determinant,(h[0][0]*rhs[1]-rhs[0]*h[1][0])/determinant]
            self.close_vectors(direct,row['dtheta_dweight'],1e-14)
            self.assertLess(row['max_abs_solve_residual'],1e-15)
            self.assertAlmostEqual(row['dquery_loss_dweight'],row['adjoint_derivative'])

    def test_every_row_parameter_and_query_derivative_against_refits(self):
        r = audit(X,Y,[1,2],1,refit=False)
        eps=1e-4
        for i,row in enumerate(r['rows']):
            wp, wm = [1.]*len(X),[1.]*len(X)
            wp[i]+=eps;wm[i]-=eps
            plus,minus=fit(X,Y,weights=wp,tolerance=1e-12),fit(X,Y,weights=wm,tolerance=1e-12)
            estimate=[(a-b)/(2*eps) for a,b in zip(plus['theta'],minus['theta'])]
            self.close_vectors(estimate,row['dtheta_dweight'],2e-8)
            lossplus=point(plus['theta'],[1,2],1)[0];lossminus=point(minus['theta'],[1,2],1)[0]
            self.assertAlmostEqual((lossplus-lossminus)/(2*eps),row['dquery_loss_dweight'],delta=2e-8)

    def test_local_approximation_error_quadratic_not_exact_deletion(self):
        r=audit(X,Y,[1,2],1,refit=False);i=5;row=r['rows'][i]
        errors=[]
        for delta in [-.1,-.05,-.025]:
            w=[1.]*len(X);w[i]+=delta
            actual=fit(X,Y,weights=w,tolerance=1e-12)['theta']
            predicted=[a+delta*b for a,b in zip(r['fit']['theta'],row['dtheta_dweight'])]
            errors.append(math.dist(actual,predicted))
        self.assertLess(errors[1],errors[0]/3)
        self.assertLess(errors[2],errors[1]/3)

    def test_finite_delete_witnesses_and_loss_nonlinearity(self):
        r=demo()['undamped'];row=r['rows'][5]
        self.assertAlmostEqual(row['predicted_delete_query_loss_delta'],.20320426327204985)
        self.assertAlmostEqual(row['actual_delete_query_loss_delta'],.30565521120251515)
        for i,row in enumerate(r['rows']):
            deleted=row['fixed_normalizer_refit']
            self.assertEqual(deleted['weights'][i],0)
            self.assertEqual(deleted['normalizer'],len(X))
            self.assertLessEqual(deleted['max_abs_gradient'],1e-10)
            self.assertAlmostEqual(point(deleted['theta'],[1,2],1)[0]-r['query_loss'],row['actual_delete_query_loss_delta'])

    def test_retained_mean_is_changed_ridge_objective(self):
        r=demo()['undamped']
        for i,row in enumerate(r['rows']):
            weights=[1.]*len(X);weights[i]=0
            equivalent=fit(X,Y,weights=weights,normalizer=len(X)-1,ridge=.1*len(X)/(len(X)-1))
            self.close_vectors(equivalent['theta'],row['fixed_normalizer_refit']['theta'],1e-8)
        self.assertGreater(abs(r['rows'][5]['retained_mean_query_loss_delta']-r['rows'][5]['actual_delete_query_loss_delta']),.01)

    def test_damping_changes_inverse_not_ridge_fit(self):
        r=demo();a,b=r['undamped'],r['damped']
        self.assertEqual(a['fit'],b['fit'])
        self.assertTrue(a['is_exact_undamped_derivative']);self.assertFalse(b['is_exact_undamped_derivative'])
        other=fit(X,Y,ridge=.6)
        self.assertNotEqual(other['theta'],a['fit']['theta'])
        self.assertGreater(abs(b['rows'][5]['predicted_delete_query_loss_delta']-a['rows'][5]['predicted_delete_query_loss_delta']),.08)
        for row in b['rows']:self.assertLess(row['max_abs_solve_residual'],1e-15)

    def test_label_complement_symmetry(self):
        a=audit(X,Y,[1,2],1);b=audit(X,[1-v for v in Y],[1,2],0)
        self.close_vectors(a['fit']['theta'],[-v for v in b['fit']['theta']])
        for u,v in zip(a['rows'],b['rows']):self.assertAlmostEqual(u['dquery_loss_dweight'],v['dquery_loss_dweight'])

    def test_row_permutation_and_orthogonal_coordinate_invariance(self):
        a=audit(X,Y,[1,2],1,refit=False)
        order=[4,1,6,0,2,5,3]
        b=audit([[X[i][1],-X[i][0]] for i in order],[Y[i] for i in order],[2,-1],1,refit=False)
        for j,i in enumerate(order):self.assertAlmostEqual(a['rows'][i]['dquery_loss_dweight'],b['rows'][j]['dquery_loss_dweight'])

    def test_duplicate_rows_are_not_collapsed(self):
        r=audit([[1],[1]],[1,1],[1],0)
        self.assertEqual(len(r['rows']),2)
        self.assertEqual(r['rows'][0]['dtheta_dweight'],r['rows'][1]['dtheta_dweight'])
        self.assertEqual(r['rows'][0]['actual_delete_query_loss_delta'],r['rows'][1]['actual_delete_query_loss_delta'])

    def test_single_row_delete_and_all_zero_weights(self):
        r=audit([[1]],[1],[1],1)
        self.assertEqual(r['rows'][0]['fixed_normalizer_refit']['theta'],[0.])
        self.assertNotIn('retained_mean_refit',r['rows'][0])
        self.assertEqual(fit(X,Y,weights=[0]*len(X))['theta'],[0.,0.])

    def test_rank_deficient_and_zero_features_have_positive_ridge_hessian(self):
        for x in [[[1,1],[1,1]],[[0,0],[0,0]]]:
            result=audit(x,[0,1],[1,1],1)
            for row in result['rows']:self.assertLess(row['max_abs_solve_residual'],1e-15)

    def test_line_search_backtracking_is_exercised(self):
        r=fit([[1,-100],[1,100],[1,99]],[0,1,0],ridge=1e-6)
        self.assertLessEqual(r['max_abs_gradient'],1e-10)
        # Alternative saturated fixture genuinely requires safeguarded Newton.
        x=[[1,36,-51],[1,-36,-92],[1,34,43],[1,-91,-4],[1,45,-86],[1,-94,10],[1,23,85],[1,-68,26],[1,-47,68],[1,31,0],[1,-96,27],[1,64,-62]]
        r=fit(x,[1,1,1,1,1,1,0,1,1,1,1,0],ridge=1e-6,weights=[.001,1,1,1,10,10,10,1,.001,.001,.1,.1])
        self.assertTrue(any(v.get('step_scale',1)<1 for v in r['history']))

    def test_seeded_systems_all_parameter_directions_and_sos_curvature(self):
        rng=random.Random(17)
        for n,d in [(3,1),(8,4),(12,6)]:
            x=[[rng.uniform(-2,2) for _ in range(d)] for _ in range(n)];y=[rng.randrange(2) for _ in range(n)];q=[rng.uniform(-1,1) for _ in range(d)]
            r=audit(x,y,q,1,refit=False)
            vector=[rng.uniform(-1,1) for _ in range(d)]
            h=r['fit']['hessian'];quad=dot(vector,[dot(row,vector) for row in h])
            sos=.1*dot(vector,vector)+math.fsum(point(r['fit']['theta'],row,label)[2]*dot(row,vector)**2/n for row,label in zip(x,y))
            self.assertAlmostEqual(quad,sos);self.assertGreaterEqual(quad,.1*dot(vector,vector)-1e-15)
            for i,row in enumerate(r['rows']):
                wp,wm=[1.]*n,[1.]*n;wp[i]+=1e-4;wm[i]-=1e-4
                p=fit(x,y,weights=wp,tolerance=1e-12)['theta'];m=fit(x,y,weights=wm,tolerance=1e-12)['theta']
                self.close_vectors([(u-v)/2e-4 for u,v in zip(p,m)],row['dtheta_dweight'],2e-8)

    def test_scalar_fit_against_independent_monotone_bisection(self):
        x=[[-2],[-1],[.5],[2]];y=[0,1,0,1];weights=[1,.3,2,1];ridge=.2
        lo,hi=-100.,100.
        for _ in range(150):
            value=(lo+hi)/2
            gradient=sum(w*((1/(1+math.exp(-value*row[0])))-label)*row[0]/4 for row,label,w in zip(x,y,weights))+ridge*value
            if gradient>0:hi=value
            else:lo=value
        result=fit(x,y,weights=weights,ridge=ridge,tolerance=1e-12)
        self.assertAlmostEqual(result['theta'][0],(lo+hi)/2,delta=1e-10)

    def test_nonconvergence_is_an_error(self):
        with self.assertRaisesRegex(ValueError,'iteration limit'):fit(X,Y,max_steps=1)
        with self.assertRaisesRegex(ValueError,'positive definite'):solve([[0]],[1])

    def test_no_mutation_rng_and_reproducibility(self):
        x,y=copy.deepcopy(X),Y[:];state=random.getstate();a=audit(x,y,[1,2],1)
        self.assertEqual(x,X);self.assertEqual(y,Y);self.assertEqual(state,random.getstate());self.assertEqual(a,audit(x,y,[1,2],1))

    def test_validation(self):
        bad=[([],[]),([[1],[2,3]],[0,1]),([[True]],[1]),([[float('nan')]],[1]),([[101]],[1]),([[1]],[True]),([[1]],[.0]),([[1]*13],[1]),([[1]]*129,[0]*129)]
        for x,y in bad:
            with self.subTest(x=x),self.assertRaises(ValueError):fit(x,y)
        for kwargs in [{'ridge':0},{'ridge':float('inf')},{'ridge':True},{'weights':[1]},{'weights':[-1]*7},{'normalizer':0},{'tolerance':0},{'max_steps':True}]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):fit(X,Y,**kwargs)
        for kwargs in [{'damping':-1},{'damping':True},{'refit':1}]:
            with self.assertRaises(ValueError):audit(X,Y,[1,2],1,**kwargs)
        with self.assertRaises(ValueError):audit(X,Y,[1],1)

    def test_cli_strict_json(self):
        base={'version':1,'X':X,'y':Y,'query':[1,2],'label':1,'ridge':.1,'damping':0,'refit':False}
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'input.json'
            path.write_text(json.dumps(base));p=subprocess.run([sys.executable,'-m','influencelab','audit',str(path)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads(p.stdout),audit(X,Y,[1,2],1,refit=False))
            cases=[dict(base,version=True),dict(base,extra=1),dict(base,label=False)]
            texts=[json.dumps(q) for q in cases]+['{"version":1,"version":1}','{"X":NaN}', ' '*100001]
            for text in texts:
                path.write_text(text);p=subprocess.run([sys.executable,'-m','influencelab','audit',str(path)],capture_output=True,text=True)
                self.assertEqual(p.returncode,2);self.assertFalse(p.stdout);self.assertIn('error:',p.stderr)

    def test_committed_demo(self):
        self.assertEqual(demo(),json.loads(Path('examples/demo.json').read_text()))

if __name__=='__main__':unittest.main()
