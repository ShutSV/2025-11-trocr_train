from transformers import AutoTokenizer
from transformers import TrOCRProcessor

tokenizer = AutoTokenizer.from_pretrained("input/trocr_cyr_processor", use_fast=False)
print(tokenizer.tokenize("Привет мир"))
print(tokenizer.tokenize("рукописный текст"))

PROCCESOR_PATH = "input/trocr_cyr_processor"
processor = TrOCRProcessor.from_pretrained(PROCCESOR_PATH, use_fast=False)
test_text = "Привет, мир"
tokens = processor.tokenizer.encode(test_text)
print(processor.tokenizer.convert_ids_to_tokens(tokens))
test_text = "рукописный текст"
tokens = processor.tokenizer.encode(test_text)
print(processor.tokenizer.convert_ids_to_tokens(tokens))
