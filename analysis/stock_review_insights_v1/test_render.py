"""Rendering safety checks: unknown values, source mismatches and sealed-run protection."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from render import render_html, render_staging, table_data


def fixture():
    windows = {day: {"equal_weight_mean": None, "positive": 0, "scored": 0, "unknown": 2} for day in ("5", "20", "60")}
    industry = {"benchmarks": [{"code": "US.TEST", "market": "US", "price_returns": {day: {"value": None} for day in windows}}],
                "groups": [{"market": "US", "group_name": "未知业务", "sample_type": "AI价值链样本", "group_id": "unknown",
                            "securities": 2, "windows": windows, "interpretation": "覆盖不足"}],
                "sector_etfs": [], "market_cutoffs": {"US": "2026-01-02"}}
    source = '<main><p>保留前文</p><h2>行业环境 · 价格证据</h2><table><tr><th>宽基/基金</th><th>5日</th><th>20日</th><th>60日</th></tr><tr><td>US.TEST</td>' + '<td>未知</td>'*3 + '</tr></table>'
    source += '<table><tr><th>AI价值链样本</th><th>5日等权均值 / 广度</th><th>20日</th><th>60日</th><th>解释</th></tr><tr><td>US · 未知业务<br><small>AI价值链样本</small></td>'
    source += '<td>未知<br><small>0/0上涨；未知2</small></td>'*3 + '<td>覆盖不足</td></tr></table><h2>保留后文</h2><a href="stocks/TEST.html">详情</a></main>'
    return source, industry


class RenderTests(unittest.TestCase):
    def test_unknowns_remain_null_and_content_outside_section_is_unchanged(self):
        source, industry = fixture()
        rendered, data = render_html(source, industry)
        self.assertTrue(rendered.startswith('<main><p>保留前文</p>'))
        self.assertTrue(rendered.endswith('<h2>保留后文</h2><a href="stocks/TEST.html">详情</a></main>'))
        self.assertEqual(data["groups"][0]["values"]["5"], {"value": None, "display": "未知", "up": 0, "scored": 0, "unknown": 2})
        self.assertIn("不可评分", rendered)
        self.assertNotIn('<script src=', rendered)
        self.assertEqual(data["market_cutoff"], {"US": "2026-01-02"})
        self.assertEqual(data["sections"][1]["id"], "US-small")

    def test_rejects_price_breadth_and_sample_mismatches(self):
        source, industry = fixture()
        with self.assertRaises(ValueError):
            table_data(source.replace('<td>未知</td>', '<td>+0.00%</td>', 1), industry)
        with self.assertRaises(ValueError):
            table_data(source.replace('0/0上涨；未知2', '0/1上涨；未知1', 1), industry)
        bad = copy.deepcopy(industry)
        bad["groups"][0]["securities"] = 3
        with self.assertRaises(ValueError):
            table_data(source, bad)

    def test_staging_refuses_sealed_or_already_rendered_report(self):
        source, industry = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "results").mkdir()
            (path / "results/industry_analysis.json").write_text(json.dumps(industry))
            (path / "index.html").write_text(source)
            (path / "manifest.json").write_text('{}')
            with self.assertRaises(ValueError):
                render_staging(path)
            self.assertEqual((path / "index.html").read_text(), source)
            (path / "manifest.json").unlink()
            render_staging(path)
            with self.assertRaises(ValueError):
                render_staging(path)


if __name__ == "__main__":
    unittest.main()
