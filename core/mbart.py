import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

article = "on the opposite side"

tokenizer.src_lang = "en_XX"
encoded_ar = tokenizer(article, return_tensors="pt")

generated_tokens = model.generate(**encoded_ar, forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"])

with torch.no_grad():
    outputs = model(**encoded_ar, output_hidden_states=True, return_dict=True)

en_hidden_states = outputs.encoder_hidden_states

de_hidden_states = outputs.decoder_hidden_states


print("encoder hidden state")
for hidden_state in en_hidden_states:
    print(hidden_state.shape)
    break

print("decoder hidden state")
for hidden_state in de_hidden_states:
    print(hidden_state.shape)
    break

#print(first_layer_activations)

answer = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

print(answer)

