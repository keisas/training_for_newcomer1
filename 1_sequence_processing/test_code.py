from unittest import TestCase
import unittest
import code
from numpy.testing import assert_array_almost_equal

class TestCode(TestCase):
  def __init__(self, *args, **kwargs):
    super(TestCode, self).__init__(*args, **kwargs)
    self.fastafile = "data/ATGCCGT.fasta"
  
  def test_base_count(self):
    self.assertEqual(code.base_count(self.fastafile), [1, 2, 2, 2],
                     msg="課題1-1, base_count()の出力が正しくありません。")

  def test_gen_rev_comp_seq(self):
    self.assertEqual(code.gen_rev_comp_seq(self.fastafile), "ACGGCAT",
                     msg="課題1-2, gen_rev_comp_seq()の出力が正しくありません。")

  def test_calc_gc_content(self):
    expected = [0.0, 50.0, 100.0, 100.0, 100.0, 50.0]
    output = code.calc_gc_content(self.fastafile, window=2, step=1)
    assert_array_almost_equal(output, expected, decimal=2, verbose=True,
                              err_msg="課題1-3, calc_gc_content()の出力が正しくありません。")

  def test_search_motif(self):
    expected = set(['R6', 'F5'])
    output = set(code.search_motif(self.fastafile, "CG"))
    self.assertEqual(output, expected,
                     msg="課題1-4, search_motif()の出力が正しくありません。") 

  def test_translate(self):
    expected = set(['MP'])
    output = set(code.translate(self.fastafile))
    self.assertEqual(output, expected,
                     msg="課題1-5, translate()の出力が正しくありません。終端コドンが無い場合は '_' を【付けない】でください。")
  

class TestALDH2(TestCase):
  def __init__(self, *args, **kwargs):
    super(TestALDH2, self).__init__(*args, **kwargs)
    self.fastafile = "data/NC_000012.fasta"
  
  def test_base_count(self):
    self.assertEqual(code.base_count(self.fastafile), [12961, 12963, 12724, 11952],
                     msg="課題1-1, base_count()の出力が正しくありません。")

  def test_gen_rev_comp_seq(self):
    "冒頭10文字だけで確認"
    self.assertEqual(code.gen_rev_comp_seq(self.fastafile)[:10], "TTTATGGTTT",
                     msg="課題1-2, gen_rev_comp_seq()の出力が正しくありません。")

  def test_calc_gc_content(self):
    "ステップが2以上の場合をテスト"
    expected = [60.0, 50.0]
    output = code.calc_gc_content(self.fastafile, window=10, step=7)[:2]
    assert_array_almost_equal(output, expected, decimal=2, verbose=True,
                              err_msg="課題1-3, calc_gc_content()の出力が正しくありません。")

  def test_search_motif_forward(self):
    expected = set(['F18188'])
    output = set(code.search_motif(self.fastafile, "ATGCTAG"))
    self.assertEqual(output, expected,
                     msg="課題1-4, search_motif()の出力が正しくありません。") 
    
  def test_search_motif_reverse(self):
    expected = set(['R18194'])
    output = set(code.search_motif(self.fastafile, "CTAGCAT"))
    self.assertEqual(output, expected,
                     msg="課題1-4, search_motif()の出力が正しくありません。") 

  def test_translate(self):
    aldh2_20letters_1 = "MLRAAARFGPRLGRRLLSAA"
    aldh2_20letters_2 = "MDASHRGRLLNRLADLIERD"
    output = set(code.translate(self.fastafile))
    output = set([x[:20] for x in output])
    judge = output & set([aldh2_20letters_1, aldh2_20letters_2])
    self.assertEqual(len(judge), 2,
                     msg="課題1-5, translate()の出力が正しくありません。")
  
if __name__ == "__main__":
  unittest.main()