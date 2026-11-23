import unittest
from datetime import datetime, timedelta, timezone

from engine import run

FLAT = (100., 100., 100., 100.)
DROP = (100., 100., 96., 97.)


def sample(bars, amounts=(100,200,300), fee=2.5, wait=1, seed=None, path="OLHC",
           initial=1000, whole=False, times=None):
    if times is None:
        start = datetime(2026,11,2,9,tzinfo=timezone.utc)
        times = [start + timedelta(minutes=5*i) for i in range(len(bars))]
    return run({"SOXL":bars,"SOXS":bars}, times, [str(t.date()) for t in times],
               {0:"SOXL"} if seed is None else seed, lambda *args: True,
               amounts, wait, None, path, fee, initial, whole)


class ScaleExecution(unittest.TestCase):
    def test_distinct_lots_and_fees(self):
        result = sample([FLAT]*6+[DROP])
        lots, s = result["lots"],result["summary"]
        self.assertEqual([t["entry_i"] for t in lots],[1,3,5])
        self.assertEqual([t["exit_i"] for t in lots],[6,6,6])
        self.assertEqual([t["net_pnl"] for t in lots],[-8.,-11.,-14.])
        self.assertEqual((s["net_pnl"],s["fees"],s["cycles"]["losses"]),(-33.,15.,1))

    def test_cancel_planned_addition_at_gap_or_intrabar_stop(self):
        for bars in ([FLAT,FLAT,FLAT,(96.,96.,96.,96.)], [FLAT,FLAT,DROP,FLAT]):
            r = sample(bars)
            self.assertEqual((r["summary"]["canceled_additions"],len(r["lots"])),(1,1))

    def test_no_opposite_cycle_while_addition_survives(self):
        # First lot takes its target at 108, while the lot entered at 105 is still open.
        # No third buy after the first exits, even if the second stays open.
        up=(105.,105.,105.,105.)
        next_bar=(105.,108.,105.,107.)
        r=sample([FLAT,FLAT,(100.,105.,100.,105.),up,next_bar,(107.,107.,107.,107.),
                  (107.,107.,107.,107.)],seed={0:"SOXL",5:"SOXS"})
        self.assertEqual([t["symbol"] for t in r["lots"]],["SOXL","SOXL"])
        self.assertEqual(r["lots"][0]["reason"],"target")
        self.assertEqual(r["lots"][1]["reason"],"open")
        self.assertEqual(r["summary"]["cycles"]["open"],1)

    def test_cash_and_integer_shares(self):
        r=sample([(80.,80.,80.,80.)]*4,amounts=(100,200),initial=245.,whole=True)
        self.assertEqual([t["qty"] for t in r["lots"]],[1,2])
        self.assertEqual(r["summary"]["ending_cash"],0.)
        self.assertEqual(r["summary"]["net_pnl"],-5.)
        r=sample([(80.,80.,80.,80.)]*4,amounts=(100,200),initial=244.99,whole=True)
        self.assertEqual((r["summary"]["cash_blocked"],len(r["lots"])),(1,1))
        r=sample([FLAT]*2,amounts=(50,),whole=True)
        self.assertEqual((r["summary"]["zero_qty_blocked"],r["summary"]["fees"]),(1,0.))

    def test_holding_with_new_week_and_no_skipped_interval_fill(self):
        start = datetime(2026,11,6,19,55,tzinfo=timezone.utc)
        times = [start, start+timedelta(days=2)]
        r=sample([FLAT,FLAT],amounts=(100,),times=times)
        self.assertEqual(r["lots"],[])

    def test_two_intrabar_paths(self):
        bars=[FLAT,(100.,108.,96.,105.)]
        self.assertEqual(sample(bars,amounts=(100,),path="OLHC")["lots"][0]["reason"],"stop")
        self.assertEqual(sample(bars,amounts=(100,),path="OHLC")["lots"][0]["reason"],"target")


if __name__ == "__main__":
    unittest.main()
