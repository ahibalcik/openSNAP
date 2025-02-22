import torch
from core.data_loader import get_dataloader
from core.models import MBartModelHandler
from core.sae import SparseAutoencoder
from core.utils import visualize_statistics, save_statistics_to_csv, Vecto2D, get_dataframe
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import random

# Set up Korean and Chinese font (e.g., Malgun Gothic)
font_path = "C:/Windows/Fonts/malgun.ttf"  # Malgun Gothic font file path
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

def main():
    # Current script directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set dataset file path (based on script directory)
    dataset_path = os.path.join(os.path.dirname(script_dir), "dataset", "simple_sentences_datasets.txt")
    
    # Load dataset
    dataloader = get_dataloader(dataset_path, MBartModelHandler().tokenizer, batch_size=32)
    print("data loader loaded")

    # Initialize mBART model
    model = MBartModelHandler()
    print("mBART model loaded")

    # Initialize SAE model
    sae_model = SparseAutoencoder()
    print("sae_model loaded")

    try:
        # Create result saving directory
        result_dir = os.path.join(script_dir, "results/result13")
        os.makedirs(result_dir, exist_ok=True)
        
        # Process data for each language
        languages = {
            'ko': 0,  # Korean
            'en': 1,  # English
            'es': 2,  # Spanish
            'zh': 3   # Chinese
        }
        
        all_vectors = []  # List to store vectors for all languages
        all_texts = []    # List to store texts for all languages
        all_lang_codes = []  # List to store language codes for all languages

        # Extract texts for each language from each batch
        vectors, all_texts, all_lang_codes = model.extract_activations(dataloader)
        # Move to GPU if available
        if torch.cuda.is_available():
            vectors = vectors.cuda()

        # Compression using SAE model
        compressed_activations, _ = sae_model(vectors)
        
        # Move to CPU and preprocess
        compressed_activations = compressed_activations.cpu().detach()

        all_vectors.append(compressed_activations)
        

        print(len(all_texts), len(all_lang_codes), len(all_vectors))
        # Combine vectors for all languages into one
        all_vectors = torch.cat(all_vectors)  # shape: (total_samples, feature_size)

        # Convert to 2D vectors (apply TSNE)
        compressed_2d = Vecto2D(all_vectors)
        # Separate results by language, express all texts in English
        all_compressed_data = []
        lang_compressed_data = {lang_code: [] for lang_code in languages.keys()}

        en_texts = [text for text, code in zip(all_texts, all_lang_codes) if code == 'en']

        for lang_code, index in languages.items():
            lang_indices = [i for i, code in enumerate(all_lang_codes) if code == lang_code]
            lang_compressed_2d = [compressed_2d[i] for i in lang_indices]
            # Express all texts in English
            lang_texts = [all_texts[i] for i in lang_indices if lang_code == lang_code]
            

            all_compressed_data.append((lang_compressed_2d, en_texts, lang_code))
            lang_compressed_data[lang_code] = (lang_compressed_2d, lang_texts, lang_code)
            print(len(lang_compressed_2d), len(lang_texts), len(lang_indices))

        small_lang_compressed_data = {lang_code: [] for lang_code in languages.keys()}
        
        
        # Set ratio for random sampling
        sample_ratio = 0.1  # e.g., 10% sampling
        num_samples = int(len(lang_compressed_data['en'][0]) * sample_ratio)
        
        # Generate random indices
        random_indices = random.sample(range(len(lang_compressed_data['en'][0])), num_samples)
        

        for lang_code in languages.keys():
            small_texts = []
            small_compressed_2d = []
            for i in random_indices:
                small_compressed_2d.append(lang_compressed_data[lang_code][0][i])
                small_texts.append(lang_compressed_data["en"][1][i])
            small_lang_compressed_data[lang_code] = (small_compressed_2d, small_texts, lang_code)
        small_compressed_data = []
        for lang_code in languages.keys():
            small_compressed_data.append(small_lang_compressed_data[lang_code])

        # Visualize and save small_compressed_data
        small_png_path = os.path.join(result_dir, f"small_statistics.png")
        visualize_statistics(small_compressed_data, save=True, filename=small_png_path)
        print(f"small statistics visualization saved to {small_png_path}")
        

        # Visualize and save statistics
        png_path = os.path.join(result_dir, "statistics.png")
        visualize_statistics(all_compressed_data, save=True, filename=png_path)
        print(f"Statistics visualization saved to {png_path}")

        # Visualize and save statistics for each language separately
        for lang_code, data in lang_compressed_data.items():
            lang_png_path = os.path.join(result_dir, f"{lang_code}_statistics.png")
            visualize_statistics([data], save=True, filename=lang_png_path)
            print(f"{lang_code} language statistics visualization saved to {lang_png_path}")
        
        # Create DataFrame and save to CSV
        data_list = []
        for compressed_2d, texts, lang_code in all_compressed_data:
            for i in range(len(compressed_2d)):
                data_dict = {
                    'x': compressed_2d[i][0],
                    'y': compressed_2d[i][1],
                    'text': texts[i],
                    'lang': lang_code
                }
                data_list.append(data_dict)

        all_df = pd.DataFrame(data_list)
        
        csv_path = os.path.join(result_dir, "statistics.csv")
        all_df.to_csv(csv_path, index=False)
        print(f"Statistics data saved to {csv_path}")
        
        
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()

