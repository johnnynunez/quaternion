import unittest
from wheel_audit import check_pe

class AuditTests(unittest.TestCase):
    def test_pe_positive_and_wrong_machine(self):
        import struct
        payload=bytearray(256); payload[:2]=b'MZ'; struct.pack_into('<I',payload,0x3c,128); payload[128:132]=b'PE\0\0'
        struct.pack_into('<H',payload,132,0xaa64)
        self.assertEqual(check_pe(bytes(payload)),0xaa64)
        struct.pack_into('<H',payload,132,0x8664)
        with self.assertRaisesRegex(ValueError,'Machine'): check_pe(bytes(payload))
        with self.assertRaisesRegex(ValueError,'PE'): check_pe(b'not a PE')

if __name__=='__main__': unittest.main()
