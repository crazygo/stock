import unittest
from price_paths import basket, daily_rows, metrics

class PricePathTests(unittest.TestCase):
    def test_trend_fit_and_efficiency_are_different(self):
        up=metrics([100,110,120,130],3)
        self.assertEqual(up['direction'],'up');self.assertAlmostEqual(up['r2'],1);self.assertAlmostEqual(up['er'],1)
        down=metrics([130,120,110,100],3)
        self.assertEqual(down['direction'],'down');self.assertAlmostEqual(down['er'],1)
        zigzag=metrics([100,110,100],2)
        self.assertEqual(zigzag['direction'],'flat');self.assertAlmostEqual(zigzag['r2'],0);self.assertAlmostEqual(zigzag['er'],0)
        flat=metrics([100,100,100],2)
        self.assertIsNone(flat['r2']);self.assertEqual(flat['er'],0)
        self.assertEqual(metrics([100,None,120],2)['status'],'missing')

    def test_normalization_does_not_weight_expensive_stocks_more(self):
        days=[f'{i:03d}' for i in range(61)]
        rows={'A':{d:[100,110,90,105] for d in days},'B':{d:[200,240,180,220] for d in days}}
        rows['A'][days[0]]=[100,100,100,100];rows['B'][days[0]]=[200,200,200,200]
        result=basket(['A','B'],rows,days)
        self.assertEqual(result['kind'],'basket_bounds')
        self.assertEqual(result['bars'][0][0],days[1])
        for actual,expected in zip(result['bars'][0][1:],[100,115,90,107.5]):self.assertAlmostEqual(actual,expected)
        del rows['B'][days[1]]
        result=basket(['A','B'],rows,days)
        self.assertEqual(result['used_ids'],['A']);self.assertEqual(result['excluded_ids'],['B'])
        self.assertEqual(result['kind'],'single_member');self.assertEqual(result['status'],'partial')
        empty=basket(['B'],rows,days)
        self.assertEqual(empty['bars'],[]);self.assertIsNone(empty['metrics']['60']['r2'])

    def test_invalid_and_duplicate_daily_bars_are_not_real_candles(self):
        self.assertEqual(daily_rows({'bars':[['d',100,90,80,100]]}),{})
        with self.assertRaises(ValueError):daily_rows({'code':'A','bars':[['d',100,110,90,100],['d',100,110,90,100]]})

    def test_group_fit_is_not_average_of_component_fits(self):
        days=[f'{i:03d}' for i in range(61)]
        a=[100+i for i in range(61)];b=[200-2*i for i in range(61)]
        rows={'A':{d:[v,v+1,v-1,v] for d,v in zip(days,a)},'B':{d:[v,v+2,v-2,v] for d,v in zip(days,b)}}
        self.assertAlmostEqual(metrics(a,60)['r2'],1);self.assertAlmostEqual(metrics(b,60)['r2'],1)
        group=basket(['A','B'],rows,days)['metrics']['60']
        self.assertEqual(group['direction'],'flat');self.assertIsNone(group['r2']);self.assertEqual(group['er'],0)

if __name__=='__main__':unittest.main()
