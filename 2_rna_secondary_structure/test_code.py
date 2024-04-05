from unittest import TestCase
import unittest
import code
from numpy.testing import assert_array_almost_equal

class TestCode(TestCase):
  def __init__(self, *args, **kwargs):
    super(TestCode, self).__init__(*args, **kwargs)
    self.fastafile = "data/AUCGCCAU.fasta"
  
  def test_enumerate_pairs(self):
    expected = set([(1, 2), (1, 8), (2, 7), (3, 4), (4, 5), (4, 6), (7, 8)])
    output = set(code.enumerate_pairs(self.fastafile))
    self.assertEqual(output, expected, 
                     msg="課題2-1, enumerate_pairs()の出力が正しくありません。i < j となるようにしてください。")

  def test_enumerate_possible_pairs(self):
    expected = set([(1, 8), (2, 7)])
    output = set(code.enumerate_possible_pairs(self.fastafile))
    self.assertEqual(output, expected,
                     msg="課題2-2, enumerate_possible_pairs()の出力が正しくありません。")

  def test_enumerate_continuous_pairs(self):
    expected = set([(1, 8, 2)])
    output = set(code.enumerate_continuous_pairs(self.fastafile))
    self.assertEqual(output, expected,
                     msg="課題2-3, enumerate_continuous_pairs()の出力が正しくありません。逆相補のペアになるので、iとjのインデックスは逆方向になります。")

  def test_create_dotbracket_notation(self):
    expected = "((....))"
    output = code.create_dotbracket_notation(self.fastafile)
    self.assertEqual(output, expected,
                     msg="課題2-4, create_dotbracket_notation()の出力が正しくありません。'('と')'の数が一致しているか確認してください。")
    
    
if __name__ == "__main__":
  unittest.main()