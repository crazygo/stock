import unittest
from financial_source_profile import profile

def amount(unit,form='20-F',filed='2026-02-01'):
    return {'units':{unit:[{'val':100,'filed':filed,'end':'2025-12-31','form':form}]}}

class SourceProfileTest(unittest.TestCase):
    def test_USD_GAAP_source_not_health(self):
        p=profile({'facts':{'us-gaap':{'Revenues':amount('USD')}}},'2026-10-02')
        self.assertTrue(p['USD_US_GAAP_field_source_presence']['revenue'])
        self.assertFalse(p['source_support_is_company_health_proof'])
    def test_foreign_USD_side_column_not_GAAP_support(self):
        p=profile({'facts':{'ifrs-full':{'Revenue':amount('USD')}}},'2026-10-02')
        self.assertEqual(p['financial_source_support'],'IFRS_taxonomy_not_supported_by_frozen_parser')
        self.assertFalse(p['USD_US_GAAP_field_source_presence']['revenue'])
    def test_EUR_GAAP_no_silent_currency_conversion(self):
        p=profile({'facts':{'us-gaap':{'Revenues':amount('EUR')}}},'2026-10-02')
        self.assertEqual(p['financial_source_support'],'non_USD_financial_currency_not_supported_by_frozen_parser')
        self.assertEqual(p['financial_reporting_units'],['EUR'])
    def test_future_disclosure_not_source_support_at_anchor(self):
        p=profile({'facts':{'us-gaap':{'Revenues':amount('USD',filed='2026-10-05')}}},'2026-10-02')
        self.assertFalse(p['USD_US_GAAP_field_source_presence']['revenue'])

if __name__=='__main__':unittest.main()
