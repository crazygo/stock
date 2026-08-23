import unittest

from enrich_industry_profit import (
    FEATURES,
    assign_profit_modules,
    extract_financial_features,
    industry_cluster,
)


def annual_fact(tag_values):
    rows = []
    for end, value, filed in tag_values:
        year = int(end[:4])
        rows.append({
            "start": f"{year-1}-01-01",
            "end": end,
            "val": value,
            "form": "10-K",
            "fp": "FY",
            "fy": year,
            "filed": filed,
        })
    return {"units": {"USD": rows}}


class IndustryProfitTests(unittest.TestCase):
    def test_industry_clusters(self):
        self.assertEqual(industry_cluster("3674"), "Electronics & Semiconductors")
        self.assertEqual(industry_cluster("7372"), "Software & Business Services")
        self.assertEqual(industry_cluster("2834"), "Chemicals & Pharmaceuticals")
        self.assertEqual(industry_cluster("6022"), "Banking & Credit")
        self.assertEqual(industry_cluster(None), "Unknown")

    def test_extract_financial_features_from_annual_companyfacts(self):
        facts = {
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": annual_fact([
                        ("2024-12-31", 1000.0, "2025-02-01"),
                        ("2025-12-31", 1200.0, "2026-02-01"),
                    ]),
                    "GrossProfit": annual_fact([
                        ("2024-12-31", 400.0, "2025-02-01"),
                        ("2025-12-31", 540.0, "2026-02-01"),
                    ]),
                    "OperatingIncomeLoss": annual_fact([
                        ("2024-12-31", 100.0, "2025-02-01"),
                        ("2025-12-31", 180.0, "2026-02-01"),
                    ]),
                    "NetIncomeLoss": annual_fact([
                        ("2024-12-31", 80.0, "2025-02-01"),
                        ("2025-12-31", 144.0, "2026-02-01"),
                    ]),
                    "NetCashProvidedByUsedInOperatingActivities": annual_fact([
                        ("2024-12-31", 150.0, "2025-02-01"),
                        ("2025-12-31", 240.0, "2026-02-01"),
                    ]),
                    "PaymentsToAcquirePropertyPlantAndEquipment": annual_fact([
                        ("2024-12-31", 50.0, "2025-02-01"),
                        ("2025-12-31", 60.0, "2026-02-01"),
                    ]),
                }
            }
        }
        result = extract_financial_features(facts)
        self.assertEqual(result["annual_period_end"], "2025-12-31")
        self.assertAlmostEqual(result["revenue_growth_yoy_pct"], 20.0)
        self.assertAlmostEqual(result["gross_margin_pct"], 45.0)
        self.assertAlmostEqual(result["operating_margin_pct"], 15.0)
        self.assertAlmostEqual(result["net_margin_pct"], 12.0)
        self.assertAlmostEqual(result["fcf_margin_pct"], 15.0)
        self.assertAlmostEqual(result["operating_margin_yoy_change_pp"], 5.0)
        self.assertEqual(result["profit_feature_count"], len(FEATURES))

    def test_profit_modules_are_data_driven_and_financials_stay_separate(self):
        rows = []
        archetypes = [
            (30, 25, 20, 18, 4),
            (5, 18, 14, 14, 0),
            (18, 4, 2, 1, 1),
            (25, -12, -15, -10, 5),
        ]
        ticker_id = 0
        for group, feature_tuple in enumerate(archetypes, start=1):
            for i in range(10):
                ticker_id += 1
                growth, op, net, fcf, change = feature_tuple
                rows.append({
                    "ticker": f"T{ticker_id}",
                    "industry_cluster": "Electronics & Semiconductors",
                    "profit_feature_count": 5,
                    "revenue_growth_yoy_pct": growth + i * 0.1,
                    "operating_margin_pct": op + i * 0.1,
                    "net_margin_pct": net + i * 0.1,
                    "fcf_margin_pct": fcf + i * 0.1,
                    "operating_margin_yoy_change_pp": change + i * 0.05,
                    "drawdown_from_peak_pct": group * 10 + i * 0.1,
                    "pullback_group": 1 + ((ticker_id - 1) % 3),
                })
        rows.append({
            "ticker": "BANK",
            "industry_cluster": "Banking & Credit",
            "profit_feature_count": 5,
            "revenue_growth_yoy_pct": 8.0,
            "operating_margin_pct": 30.0,
            "net_margin_pct": 20.0,
            "fcf_margin_pct": 10.0,
            "operating_margin_yoy_change_pp": 2.0,
            "drawdown_from_peak_pct": 20.0,
            "pullback_group": 2,
        })
        result = assign_profit_modules(rows)
        self.assertGreaterEqual(result["k"], 4)
        self.assertGreater(result["fit_count"], 30)
        self.assertEqual(rows[-1]["profit_module"], "P-FIN 金融口径单列")
        normal_modules = {row["profit_module"] for row in rows[:-1]}
        self.assertGreaterEqual(len(normal_modules), 4)


if __name__ == "__main__":
    unittest.main()
