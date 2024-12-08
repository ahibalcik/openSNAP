# core/models.py

import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from torch.utils.data import DataLoader

class MBartModelHandler:
    def __init__(self):
        """
        Initialize mBART-50 model and tokenizer
        """
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def extract_activations(self, dataloader: DataLoader):
        """
        Extract hidden layer activations from input text using DataLoader
        
        Args:
            dataloader: DataLoader instance of TextDataset
            
        Returns:
            tuple: (combined_pooled_batch, texts)
                - combined_pooled_batch: Combined encoder-decoder vectors in batch
                - texts: List of original input texts
        """
        self.model.eval()
        combined_pooled_batch = []
        texts = []
        lang_codes = []
        
        print("Starting activation extraction")
        total_batches = len(dataloader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader, 1):
                # Extract texts from batch
                for idx, item in enumerate(batch):
                    if idx == 0:
                        ko_texts = item  # Korean texts
                        texts.extend(ko_texts)
                        lang_codes.extend(['ko'] * len(ko_texts))
                        #print(ko_texts)
                    elif idx == 1:
                        en_texts = item  # English texts
                        texts.extend(en_texts)
                        lang_codes.extend(['en'] * len(en_texts))
                        #print(en_texts)
                    elif idx == 2:
                        es_texts = item  # Spanish texts
                        texts.extend(es_texts)
                        lang_codes.extend(['es'] * len(es_texts))
                        #print(es_texts)
                    elif idx == 3:
                        zh_texts = item  # Chinese texts
                        texts.extend(zh_texts)
                        lang_codes.extend(['zh'] * len(zh_texts))
                        #print(zh_texts)
                
                # Progress update
                print(f"Processing batch {batch_idx}/{total_batches}", end='\r')
                
                # Process each language
                for lang_text, lang_code in zip([ko_texts, en_texts, es_texts, zh_texts], ['ko', 'en', 'es', 'zh']):
                    for text in lang_text:
                        self.tokenizer.src_lang = lang_code
                        encoded_batch = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                        
                        outputs = self.model(**encoded_batch, output_hidden_states=True, return_dict=True)
                        
                        # Extract and process hidden states
                        encoder_last_hidden_state = outputs.encoder_last_hidden_state
                        decoder_first_hidden_state = outputs.decoder_hidden_states[1]
                        
                        encoder_mean_pooled = torch.mean(encoder_last_hidden_state, dim=1)
                        decoder_mean_pooled = torch.mean(decoder_first_hidden_state, dim=1)
                        
                        combined_pooled = encoder_mean_pooled + decoder_mean_pooled
                        combined_pooled_batch.append(combined_pooled)
        
        final_vectors = torch.cat(combined_pooled_batch, dim=0)
        print(f"\nExtraction complete. Processed {len(texts)} texts, vector shape: {final_vectors.shape}")
        
        return final_vectors, texts, lang_codes

if __name__ == "__main__":
    from data_loader import get_dataloader
    
    model = MBartModelHandler()
    dataloader = get_dataloader("simple_sentences_datasets.txt", model.tokenizer, batch_size=32)
    vectors, texts = model.extract_activations(dataloader)
    print(f"Extracted vector shape: {vectors.shape}")
    print(f"Extracted texts: {texts}")