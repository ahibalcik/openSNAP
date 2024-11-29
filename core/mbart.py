import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
article = "사과"

tokenizer.src_lang = "ko_KR"
encoded_ar = tokenizer(article, return_tensors="pt")

generated_tokens = model.generate(**encoded_ar, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])

with torch.no_grad():
    outputs = model(**encoded_ar, output_hidden_states=True, return_dict=True)

hidden_states = outputs.decoder_hidden_states

first_layer_activations = hidden_states[1:5]

print(first_layer_activations)

answer = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

print(answer)

