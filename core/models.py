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

    def extract_activations(self, dataloader: DataLoader, src_lang: str):
        """
        Extract hidden layer activations from input text using DataLoader
        
        Args:
            dataloader: DataLoader instance of TextDataset
            src_lang: Source language code (e.g., "ko_KR")
            
        Returns:
            tuple: (combined_pooled_batch, texts)
                - combined_pooled_batch: Combined encoder-decoder vectors in batch
                - texts: List of original input texts
        """
        self.model.eval()
        combined_pooled_batch = []
        texts = []
        
        print(f"Starting activation extraction for {src_lang}")
        total_batches = len(dataloader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader, 1):
                # Extract texts from batch
                ko_texts = [item[0] for item in batch]
                texts.extend(ko_texts)
                
                # Progress update
                print(f"Processing batch {batch_idx}/{total_batches}", end='\r')
                
                # Process batch
                self.tokenizer.src_lang = src_lang
                encoded_batch = self.tokenizer(ko_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                
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
        
        return final_vectors, texts

if __name__ == "__main__":
    from data_loader import get_dataloader
    
    model = MBartModelHandler()
    dataloader = get_dataloader("simple_sentences_datasets.txt", model.tokenizer, batch_size=32)
    vectors, texts = model.extract_activations(dataloader, "ko_KR")
    print(f"Extracted vector shape: {vectors.shape}")
    print(f"Extracted texts: {texts}")