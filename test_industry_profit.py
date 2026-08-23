import unittest

from enrich_industry_profit import (
    FEATURES,
    add_neutral_residuals,
    assign_profit_modules,
    normalize_yahoo_info,
)


class IndustryProfitTests(unittest.TestCase):
    def test_normalize_yahoo_info(self):
        info = {
            "sector": "Technology",
            "industry": "Semiconductors",
            "country": "United States",
            "marketCap": 1_000_000_000,
            "totalRevenue": 1_200.0,
            "freeCashflow": 180.0,
            "revenueGrowth": 0.20,
            "operatingMargins": 0.15,
            "profitMargins": 0.12,
            "grossMargins": 0.45,
            "earningsGrowth": 0.30,
            "returnOnEquity": 0.25,
        }
        result = normalize_yahoo_info(info)
        self.assertEqual(result["sector"], "Technology")
        self.assertEqual(result["industry"], "Semiconductors")
        self.assertAlmostEqual(result["revenue_growth_yoy_pct"], 20.0)
        self.assertAlmostEqual(result["gross_margin_pct"], 45.0)
        self.assertAlmostEqual(result["operating_margin_pct"], 15.0)
        self.assertAlmostEqual(result["net_margin_pct"], 12.0)
        self.assertAlmostEqual(result["fcf_margin_pct"], 15.0)
        self.assertAlmostEqual(result["earnings_growth_yoy_pct"], 30.0)
        self.assertAlmostEqual(result["return_on_equity_pct"], 25.0)
        self.assertEqual(result["profit_feature_count"], len(FEATURES))
        self.assertEqual(result["fundamentals_status"], "ok")

    def test_profit_modules_are_data_driven_and_financials_stay_separate(self):
        rows = []
        archetypes = [
            (30, 25, 20, 18, 35),
            (5, 18, 14, 14, 5),
            (18, 4, 2, 1, 15),
            (25, -12, -15, -10, 20),
        ]
        ticker_id = 0
        for group, feature_tuple in enumerate(archetypes, start=1):
            for i in range(10):
                ticker_id += 1
                growth, op, net, fcf, earnings = feature_tuple
                rows.append({
                    "ticker": f"T{ticker_id}",
                    "sector": "Technology",
                    "industry": "Semiconductors",
                    "profit_feature_count": 5,
                    "revenue_growth_yoy_pct": growth + i * 0.1,
                    "operating_margin_pct": op + i * 0.1,
                    "net_margin_pct": net + i * 0.1,
                    "fcf_margin_pct": fcf + i * 0.1,
                    "earnings_growth_yoy_pct": earnings + i * 0.1,
                    "drawdown_from_peak_pct": group * 10 + i * 0.1,
                    "pullback_group": 1 + ((ticker_id - 1) % 3),
                })
        rows.append({
            "ticker": "BANK",
            "sector": "Financial Services",
            "industry": "Banks - Regional",
            "profit_feature_count": 5,
            "revenue_growth_yoy_pct": 8.0,
            "operating_margin_pct": 30.0,
            "net_margin_pct": 20.0,
            "fcf_margin_pct": 10.0,
            "earnings_growth_yoy_pct": 2.0,
            "drawdown_from_peak_pct": 20.0,
            "pullback_group": 2,
        })
        result = assign_profit_modules(rows)
        self.assertGreaterEqual(result["k"], 4)
        self.assertGreater(result["fit_count"], 30)
        self.assertEqual(rows[-1]["profit_module"], "P-FIN 金融口径单列")
        normal_modules = {row["profit_module"] for row in rows[:-1]}
        self.assertGreaterEqual(len(normal_modules), 4)

    def test_industry_residual_falls_back_to_sector_when_peer_group_small(self):
        rows = []
        for i in range(6):
            rows.append({
                "ticker": f"A{i}",
                "sector": "Technology",
                "industry": "Semiconductors",
                "profit_module": "P1 高增长高盈利",
                "drawdown_from_peak_pct": 10.0 + i,
            })
        rows.append({
            "ticker": "SMALL",
            "sector": "Technology",
            "industry": "Tiny Industry",
            "profit_module": "P1 高增长高盈利",
            "drawdown_from_peak_pct": 30.0,
        })
        add_neutral_residuals(rows)
        small = rows[-1]
        self.assertTrue(small["industry_benchmark_basis"].startswith("sector_fallback:"))
        self.assertIsNotNone(small["industry_adjusted_drawdown_residual_pp"])
        self.assertTrue(rows[0]["peer_benchmark_basis"].startswith("industry×module:"))


if __name__ == "__main__":
    unittest.main()
