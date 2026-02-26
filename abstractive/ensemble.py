
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

class AbstractiveEnsemble:
    # mBART-50 language code mapping
    MBART_LANG_MAP = {
        'en': 'en_XX',
        'gu': 'gu_IN',
        'hi': 'hi_IN',
        'bn': 'bn_IN',
        'ta': 'ta_IN',
        'te': 'te_IN',
        'ml': 'ml_IN',
        'mr': 'mr_IN',
        'ne': 'ne_NP',
        'ur': 'ur_PK',
    }

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.model_names = {
            'bart': 'facebook/mbart-large-50-many-to-many-mmt',
            't5': 'csebuetnlp/mT5_multilingual_XLSum',
            'pegasus': 'google/pegasus-cnn_dailymail'
        }

    def load_model(self, model_key):
        """
        Loads a specific model and tokenizer.
        """
        if model_key in self.models:
            return self.models[model_key]
        
        print(f"Loading {model_key} model...")
        try:
            model_name = self.model_names[model_key]
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Move model to the selected device
            model = model.to(self.device)
            
            # Store as tuple
            self.models[model_key] = (tokenizer, model)
            return (tokenizer, model)
        except Exception as e:
            print(f"Error loading {model_key}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_candidates(self, text, max_length=150, min_length=30, language='en'):
        """
        Generates summaries from all available models.
        """
        candidates = {}

        for key in self.model_names.keys():
            candidate = self.generate_single_candidate(text, key, max_length, min_length, language=language)
            if candidate:
                candidates[key] = candidate

        return candidates

    def get_available_models(self):
        """Returns list of available model keys."""
        return list(self.model_names.keys())

    def generate_single_candidate(self, text, model_key, max_length=150, min_length=30, language='en'):
        """
        Generates a summary from a specific model.
        Handles mBART language codes for 'bart', standard generation for others.
        """
        loaded = self.load_model(model_key)
        if not loaded:
            return ""

        tokenizer, model = loaded
        try:
            inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
            input_ids = inputs.input_ids.to(self.device)

            generate_kwargs = {
                "max_length": max_length,
                "min_length": min_length,
                "num_beams": 4,
                "early_stopping": True,
            }

            # mBART requires special language token handling
            if model_key == 'bart' and hasattr(tokenizer, 'lang_code_to_id'):
                mbart_lang = self._get_mbart_lang_code(language)
                tokenizer.src_lang = mbart_lang
                # Re-tokenize with correct source language
                inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
                input_ids = inputs.input_ids.to(self.device)
                generate_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id.get(
                    mbart_lang, tokenizer.lang_code_to_id.get('en_XX')
                )

            summary_ids = model.generate(input_ids, **generate_kwargs)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            print(f"Error generating summary with {model_key}: {e}")
            import traceback
            traceback.print_exc()
            return ""

if __name__ == "__main__":
    # Test block
    ensemble = AbstractiveEnsemble()
    
    sample_text = """
    The Amazon rainforest, covering much of northwestern Brazil and extending into Colombia, Peru and other South American countries, is the world’s largest tropical rainforest, famed for its biodiversity. It’s crisscrossed by thousands of rivers, including the powerful Amazon. River towns, with 19th-century architecture from rubber-boom days, include Brazil’s Manaus and Belém. Ecuador’s Amazon region, reachable from its capital Quito, is known for its rich wildlife including colorful toucans, macaws and spider monkeys.
    """
    
    print("Generating summaries...")
    results = ensemble.generate_candidates(sample_text)
    
    for model, summary in results.items():
        print(f"\n[{model.upper()}]:\n{summary}")
