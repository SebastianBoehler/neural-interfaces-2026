"""The complete report must preserve every arm/order and serialize as JSON."""
import json
import unittest
import numpy as np
from report import build_report


class ReportTest(unittest.TestCase):
 def test_complete_report_has_nested_json_keys_and_paired_contrasts(self):
  orders=('within0','within1','within2','within3','full0','full1','runs0','runs1')
  people=tuple(f'P{i:02d}' for i in range(1,11));arms=('TF','TFD','reference')
  official={(a,o):float(i+1)*10 for i,a in enumerate(arms) for o in orders}
  pp={(a,o):np.full(10,float(i+1)*10) for i,a in enumerate(arms) for o in orders}
  def summarize(d,order,persons):return {'mean':float(np.mean(d))}
  result=build_report(official,pp,orders,people,{'fold.npz':'abc'},'source',summarize)
  self.assertEqual(set(result['official_equal_cell_by_order']),set(arms))
  self.assertEqual(result['official_equal_cell_by_order']['TF']['within0'],10.)
  self.assertEqual(result['official_equal_cell_scheme_equal']['TFD'],20.)
  self.assertEqual(result['people_order_intervals']['TFD_minus_TF']['mean'],10.)
  json.dumps(result,allow_nan=False)


if __name__=='__main__':unittest.main()

class ActualSourceTest(unittest.TestCase):
 def test_existing_collector_patch_compiles(self):
  from pathlib import Path
  from score_fixed import patch
  path=Path(__file__).resolve().parent.parent/'score.py'
  fixed=patch(path.read_text())
  compile(fixed,str(path),'exec')
  self.assertIn('report=build_report(official,pp,ORDERS,PEOPLE,folds,',fixed)
