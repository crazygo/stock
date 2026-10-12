import json,unittest
import numpy as np
from run_joint_v2 import json_ready


class RecordTests(unittest.TestCase):
    def test_probabilities_roundtrip_without_rounding_or_mutation(self):
        p=np.array([.1234567890123456,.2,.3,.3765432109876544])
        record={'raw':p,'nested':{'adjusted':p.copy()},'missing':None}
        encoded=json.dumps(json_ready(record));decoded=json.loads(encoded)
        self.assertIs(record['raw'],p)
        np.testing.assert_array_equal(decoded['raw'],p)
        np.testing.assert_array_equal(decoded['nested']['adjusted'],p)
        self.assertIsNone(decoded['missing'])


if __name__=='__main__':unittest.main()
