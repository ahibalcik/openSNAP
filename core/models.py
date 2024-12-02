# core/models.py

import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

class MBartModelHandler:
    def __init__(self):
        """
        Initialize the mBART-50 model and tokenizer.
        """
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        

    def extract_activations(self, input_text: str, src_lang: str):
        """
        Extract hidden layer activations for a given input text.
        """
        self.tokenizer.src_lang = "ko_KR"
        encoded_ar = self.tokenizer(input_text, return_tensors="pt")

        generated_tokens = self.model.generate(**encoded_ar, forced_bos_token_id=self.tokenizer.lang_code_to_id["en_XX"])

        with torch.no_grad():
            outputs = self.model(**encoded_ar, output_hidden_states=True, return_dict=True)  

        # Extract encoder and decoder last hidden states
        encoder_last_hidden_state = outputs.encoder_last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]
        decoder_first_hidden_state = outputs.decoder_hidden_states[1]  # Shape: [batch_size, seq_len, hidden_dim]

        # Perform MeanPooling on the hidden states along the sequence length
        # MeanPooling reduces seq_len dimension by taking minimum along dim=1
        encoder_mean_pooled = torch.mean(encoder_last_hidden_state, dim=1).values  # Shape: [batch_size, hidden_dim]
        decoder_mean_pooled = torch.mean(decoder_first_hidden_state, dim=1).values  # Shape: [batch_size, hidden_dim]

        combined_pooled = encoder_mean_pooled + decoder_mean_pooled # Element-wise sum

        return combined_pooled