from transformers import AutoConfig, AutoModelForCausalLM
from .cvlm import CvlmForCausalLM, CvlmConfig

AutoConfig.register("cvlm", CvlmConfig)
AutoModelForCausalLM.register(CvlmConfig, CvlmForCausalLM)
