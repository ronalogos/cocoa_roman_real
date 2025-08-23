from cobaya.likelihoods.roman_real._cosmolike_prototype_base import _cosmolike_prototype_base, survey
import cosmolike_roman_real_interface as ci
import numpy as np

class roman_real_cosmic_shear(_cosmolike_prototype_base):
  def initialize(self):
    super(roman_real_cosmic_shear,self).initialize(probe="xi")