"""Real directory edge cases and conservative security-type boundaries."""
import unittest
from catalog import classify

class CatalogTest(unittest.TestCase):
    def row(self,name,ticker='TEST',**flags):
        return {'Symbol':ticker,'Security Name':name,'ETF':'N','Test Issue':'N',**flags}
    def test_equity_security_descriptors(self):
        for name in ['Evaxion A/S - American Depositary Share','Alibaba - American Depositary Shares',
                     'Silence - American Depository Share','ASML - New York Registry Shares',
                     'Logitech - Registered Shares','Shopify - Class A Subordinate Voting Shares',
                     'Alphabet - Class C Capital Stock','GVH - Ord Shares','TNL Common  Stock']:
            with self.subTest(name=name):self.assertEqual(classify(self.row(name))[0],'candidate')
    def test_security_not_issuer_name(self):
        self.assertEqual(classify(self.row('Preferred Bank - Common Stock'))[0],'candidate')
    def test_derivative_overrides_embedded_common(self):
        for name in ['Acquisition Corp - Unit','Corp - Right','Corp - Units including Common Stock',
                     'Corp - Preferred Share','Corp - Depositary Shares representing Preferred Stock',
                     'ETRACS Index ETN','Corp - Warrants to purchase Common Stock','Corp - Closed End Fund']:
            with self.subTest(name=name):self.assertEqual(classify(self.row(name))[0],'excluded')
    def test_uncertain_retained(self):
        self.assertEqual(classify(self.row('Taiwan Semiconductor Manufacturing Company Ltd.'),True)[0],'review')
        self.assertEqual(classify(self.row('Cadiz - Depositary Shares'),True)[0],'review')
        self.assertEqual(classify(self.row('Bank Nova Scotia Halifax Pfd 3 Ordinary Shares','BNS'),True)[0],'review')
    def test_directory_flags_override(self):
        self.assertEqual(classify(self.row('Common Stock',ETF='Y'))[0],'excluded')
        self.assertEqual(classify(self.row('Common Stock',**{'Test Issue':'Y'}))[0],'excluded')
        self.assertEqual(classify(self.row('Bare series','DBRG$H'),True)[0],'excluded')

if __name__=='__main__':unittest.main()
