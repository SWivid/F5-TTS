import re
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import random


class MongolianTextProcessor:
    def __init__(self, seed: int = 42):
        
        # Set random seed for reproducibility at the start
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Correct Mongolian Cyrillic categorization
        self.mongolian_vowels = set('аэиоуөүяеёюйы')
        self.mongolian_consonants = set('бвгджзклмнпрстфхцчшщ')
        self.mongolian_modifiers = set('ъь')  # Special modifiers
        
        self.phoneme_weights = self._initialize_phoneme_weights()
        
        # Vowel harmony groups (important for pronunciation)
        self.masculine_vowels = set('аоуяё')
        self.feminine_vowels = set('эөүеи') 
        self.neutral_vowels = set('ий')      # Neutral vowels
        
        # Common diphthongs and important combinations
        self.diphthongs = {
            # й-ending diphthongs
            'ай', 'эй', 'ой', 'уй', 'үй', 'яй', 'ёй', 'юй',
            # Long vowels
            'уу', 'үү', 'юу', 'юү', 'яу', 'ёу', 'еү',
            # Special cases
            'и', 'ий', 'ы', 'эй'
        }
        
        # Initialize English phoneme system
        self.english_pattern = re.compile(r'\b[a-zA-Z]+\b')
        self.english_phonemes = self._load_english_phonemes()
        self.english_phoneme_mapping = self._initialize_english_phoneme_mapping()
        
        # Add syllable patterns
        self.syllable_patterns = {
            'V': r'[аэиоуөүяеёюйы]',
            'C': r'[бвгджзклмнпрстфхцчшщ]',
            'M': r'[ъь]'  # Modifiers
        }
        
        # Define valid syllable structures
        self.valid_syllables = [
            'V', 'CV', 'VC', 'CVC',
            'VV', 'CVV', 'CVVC', 'CCVC'
        ]
        
        # Position embeddings per syllable position
        self.syllable_position_embeddings = self._initialize_position_embeddings(dim=512)

        
        # Add pause and emphasis tokens
        self.LONG_PAUSE = "<long_pause>"  # For -- and similar
        self.SHORT_PAUSE = "<short_pause>"  # For commas, natural breaks
        self.EMPHASIS_START = "<emphasis>"
        self.EMPHASIS_END = "</emphasis>"
        self.QUESTION_START = "<question>"
        self.QUESTION_END = "</question>"
        self.EXCLAMATION_START = "<exclaim>"
        self.EXCLAMATION_END = "</exclaim>"
        
        # Abbreviation patterns (both Mongolian and English)
        self.abbreviation_pattern = re.compile(
            r'\b[А-ЯӨҮa-zA-Z]{2,}\.?\b'  # Catches both ҮАБТХ and NATO
        )
        
        # Punctuation and pause patterns
        self.dash_pattern = re.compile(r'--+|—|–')  # Various dash types
        self.quote_pattern = re.compile(r'["\'"](.*?)[\'""]')
        self.special_punct_pattern = re.compile(r'[!?]+')
        
        self.numeric_pattern = re.compile(r'\d+')
        self.roman_pattern = re.compile(
            r'\b(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b'
        )
        
        self.number_converter = MongolianNumberConverter()
        
        self.special_tokens = set([
            self.LONG_PAUSE, self.SHORT_PAUSE,
            self.EMPHASIS_START, self.EMPHASIS_END,
            self.QUESTION_START, self.QUESTION_END,
            self.EXCLAMATION_START, self.EXCLAMATION_END,
            '<letter_mn>', '</letter_mn>',
            '<letter_en>', '</letter_en>',
            '<english>', '</english>'
        ])
        
        # Common Mongolian suffixes and their variations
        self.suffix_patterns = {
            # Possessive suffixes
            'тай/тэй/той/төй': {
                'masculine': {'а': 'тай', 'о': 'той'},
                'feminine': {'э': 'тэй', 'ө': 'төй'},
            },
            # Instrumental case
            'аар/ээр/оор/өөр': {
                'masculine': {'а': 'аар', 'о': 'оор'},
                'feminine': {'э': 'ээр', 'ө': 'өөр'}
            },
            # Dative/locative case
            'д/т': {
                'masculine': 'д',
                'feminine': 'т'
            },
            # Ablative case
            'аас/ээс/оос/өөс': {
                'masculine': {'а': 'аас', 'о': 'оос'},
                'feminine': {'э': 'ээс', 'ө': 'өөс'}
            },
            # Plural suffixes
            'ууд/үүд': {
                'masculine': 'ууд',
                'feminine': 'үүд'
            },
            'нууд/нүүд': {
                'masculine': 'нууд',
                'feminine': 'нүүд'
            },
            # Direction/motion suffixes
            'руу/рүү': {
                'masculine': 'руу',
                'feminine': 'рүү'
            },
            # Reflexive suffixes
            'аа/ээ/оо/өө': {
                'masculine': {'а': 'аа', 'о': 'оо'},
                'feminine': {'э': 'ээ', 'ө': 'өө'}
            }
        }
        
        # Special cases and exceptions
        self.special_cases = {
            'гэр': 'feminine',  # Always feminine despite 'э'
            'нар': 'masculine', # Always masculine
            'бэр': 'feminine',
            'хэр': 'feminine'
        }
        
        # Compound word connectors
        self.compound_connectors = set(['-', '‐', '᠆']) 
        
    def is_special_token(self, token: str) -> bool:
        """Helper to check if a token is special"""
        return token in self.special_tokens

    def get_token_length(self, token: str) -> int:
        """Get the effective length of a token"""
        return 1 if self.is_special_token(token) else len(token)
        
    def convert_numbers(self, text: str) -> str:
        """Convert numbers and Roman numerals to Mongolian text"""
        def replace_number(match):
            number = int(match.group(0))
            return self.number_converter.convert_to_words(number)
            
        def replace_roman(match):
            roman = match.group(0)
            return self.number_converter.convert_roman_to_mongolian(roman)
        
        # Convert regular numbers
        text = self.numeric_pattern.sub(replace_number, text)
        # Convert Roman numerals
        text = self.roman_pattern.sub(replace_roman, text)
        
        return text
        
    def _initialize_position_embeddings(self, dim: int) -> Dict[str, torch.Tensor]:
        """Initialize position embeddings with fixed seed"""
        embeddings = {}
        for pos in ['initial', 'medial', 'final', 'isolated']:
            # Use nn.Parameter for proper model integration
            param = nn.Parameter(torch.zeros(dim))
            # Initialize with fixed values
            nn.init.normal_(param, mean=0.0, std=0.02)
            embeddings[pos] = param
        return embeddings
        
    def _load_english_phonemes(self) -> Dict[str, List[str]]:
        """Load CMU dictionary for English pronunciation"""
        import nltk
        try:
            from nltk.corpus import cmudict
            nltk.data.find('corpora/cmudict')
        except LookupError:
            nltk.download('cmudict')
        
        pronunciation_dict = cmudict.dict()
        
        # Process into more usable format
        phoneme_dict = {}
        for word, pronunciations in pronunciation_dict.items():
            # Take first pronunciation if multiple exist
            if pronunciations:
                phoneme_dict[word] = pronunciations[0]
        
        return phoneme_dict

    def _initialize_english_phoneme_mapping(self) -> Dict[str, List[str]]:
        """Map English phonemes to closest Mongolian sounds"""
        return {
            'AA': 'а',  # as in "odd"
            'AE': 'э',  # as in "at"
            'AH': 'а',  # as in "hut"
            'AO': 'о',  # as in "ought"
            'AW': 'ау', # as in "cow"
            'AY': 'ай', # as in "hide"
            'B': 'б',   # as in "bee"
            'CH': 'ч',  # as in "cheese"
            'D': 'д',   # as in "dee"
            'DH': 'д',  # as in "thee"
            'EH': 'э',  # as in "Ed"
            'ER': 'өр', # as in "hurt"
            'EY': 'эй', # as in "ate"
            'F': 'ф',   # as in "fee"
            'G': 'г',   # as in "green"
            'HH': 'х',  # as in "he"
            'IH': 'и',  # as in "it"
            'IY': 'ий', # as in "eat"
            'JH': 'ж',  # as in "gee"
            'K': 'к',   # as in "key"
            'L': 'л',   # as in "lee"
            'M': 'м',   # as in "me"
            'N': 'н',   # as in "knee"
            'NG': 'нг', # as in "ping"
            'OW': 'оу', # as in "oat"
            'OY': 'ой', # as in "toy"
            'P': 'п',   # as in "pee"
            'R': 'р',   # as in "read"
            'S': 'с',   # as in "sea"
            'SH': 'ш',  # as in "she"
            'T': 'т',   # as in "tea"
            'TH': 'т',  # as in "theta"
            'UH': 'ү',  # as in "hood"
            'UW': 'у',  # as in "two"
            'V': 'в',   # as in "vee"
            'W': 'в',   # as in "we"
            'Y': 'й',   # as in "yield"
            'Z': 'з',   # as in "zee"
            'ZH': 'ж',  # as in "seizure"
        }

    def _initialize_phoneme_weights(self) -> Dict[str, float]:
        """Initialize weights for important phoneme combinations"""
        weights = {}
        
        # Diphthongs with й
        й_diphthongs = ['ай', 'эй', 'ой', 'уй', 'үй', 'яй', 'ёй', 'юй']
        for dip in й_diphthongs:
            weights[dip] = 1.5  # These need special attention for proper pronunciation
            
        # Long vowels
        long_vowels = ['уу', 'үү', 'юу', 'юү', 'яу', 'ёу', 'еү']
        for vowel in long_vowels:
            weights[vowel] = 1.8  # Length distinction is crucial
            
        # Special cases
        special_cases = {
            'и': 1.3,
            'ий': 1.5,
            'ы': 1.4,
            'эй': 1.5
        }
        weights.update(special_cases)
        
        # Consonant combinations that need special attention
        consonant_combinations = ['нг', 'кс', 'пт', 'рт', 'лт', 'ст']
        for combo in consonant_combinations:
            weights[combo] = 1.4
            
        return weights

    def handle_english_words(self, text: str) -> str:
        def convert_to_mongolian_phonemes(match):
            word = match.group(0).lower()
            if word in self.english_phonemes:
                phonemes = self.english_phonemes[word]
                mongolian_sounds = []
                
                for phoneme in phonemes:
                    # Remove stress markers (numbers) from CMU phonemes
                    base_phoneme = ''.join([c for c in phoneme if not c.isdigit()])
                    
                    # Get corresponding Mongolian sound
                    if base_phoneme in self.english_phoneme_mapping:
                        mongolian_sounds.append(self.english_phoneme_mapping[base_phoneme])
                
                # Mark as English word for special handling
                return f"<english>{''.join(mongolian_sounds)}</english>"
            
            return word  # Fallback to original word if not found

        return self.english_pattern.sub(convert_to_mongolian_phonemes, text)
        
    def get_vowel_class(self, word: str) -> str:
        """Determine vowel class of a word with special case handling"""
        # Check special cases first
        if word.lower() in self.special_cases:
            return self.special_cases[word.lower()]
            
        # Handle compound words
        if any(conn in word for conn in self.compound_connectors):
            # For compounds, use the last part for harmony
            parts = [p for p in re.split(f'[{"".join(self.compound_connectors)}]', word) if p]
            if parts:
                word = parts[-1]
        
        # Find first meaningful vowel
        vowels_in_word = []
        for char in word.lower():
            if char in self.masculine_vowels:
                vowels_in_word.append('masculine')
            elif char in self.feminine_vowels:
                vowels_in_word.append('feminine')
            elif char in self.neutral_vowels:
                continue  # Skip neutral vowels
                
        # Decision logic
        if vowels_in_word:
            # Use the first non-neutral vowel
            return vowels_in_word[0]
        return 'neutral'

    def harmonize_suffix(self, stem: str, suffix: str) -> str:
        """Harmonize a suffix with its stem"""
        vowel_class = self.get_vowel_class(stem)
        
        # Handle suffixes that match known patterns
        for pattern, harmonies in self.suffix_patterns.items():
            pattern_parts = pattern.split('/')
            if suffix in pattern_parts:
                if vowel_class == 'neutral':
                    return suffix
                # Direct match with a pattern part
                if isinstance(harmonies[vowel_class], str):
                    return harmonies[vowel_class]
                else:
                    # Need to check dominant vowel
                    dominant_vowel = self.get_dominant_vowel(stem)
                    return harmonies[vowel_class].get(dominant_vowel, suffix)
        
        # Handle general vowel harmonization for unknown suffixes
        return self.general_vowel_harmony(suffix, vowel_class)

    def get_dominant_vowel(self, word: str) -> str:
        """Get the dominant vowel type (а/э/о/ө) in a word"""
        vowel_counts = {'а': 0, 'э': 0, 'о': 0, 'ө': 0}
        for char in word.lower():
            if char in vowel_counts:
                vowel_counts[char] += 1
                
        # Return the most frequent vowel, with priority to first occurrence if tied
        max_count = 0
        dominant = 'а'  # default
        for vowel in reversed(['а', 'э', 'о', 'ө']):  # Reverse to give priority to first occurrence
            if vowel_counts[vowel] > max_count:
                max_count = vowel_counts[vowel]
                dominant = vowel
        return dominant

    def general_vowel_harmony(self, text: str, vowel_class: str) -> str:
        """Apply general vowel harmony rules to unknown text"""
        vowel_pairs = [
            ('а', 'э'),
            ('о', 'ө'),
            ('у', 'ү'),
            ('я', 'е'),
            ('ё', 'е')
        ]
        
        result = text
        if vowel_class == 'masculine':
            for fem, masc in vowel_pairs:
                result = result.replace(fem, masc)
        elif vowel_class == 'feminine':
            for masc, fem in vowel_pairs:
                result = result.replace(masc, fem)
                
        return result

    def apply_vowel_harmony(self, text: str) -> str:
        """Apply vowel harmony rules to text"""
        words = text.split()
        harmonized_words = []
        
        for word in words:
            # Skip words without suffixes
            if not any(conn in word for conn in self.compound_connectors):
                harmonized_words.append(word)
                continue
                
            # Split into parts (stem and suffixes)
            parts = [p for p in re.split(f'[{"".join(map(re.escape, self.compound_connectors))}]', word) if p]
            if len(parts) < 2:
                harmonized_words.append(word)
                continue
                
            # Get stem and suffixes
            stem = parts[0]
            suffixes = parts[1:]
            
            # Harmonize each suffix
            harmonized_suffixes = [
                self.harmonize_suffix(stem, suffix)
                for suffix in suffixes
            ]
            
            # Reconstruct word
            harmonized_word = stem + '-' + '-'.join(harmonized_suffixes)
            harmonized_words.append(harmonized_word)
        
        return ' '.join(harmonized_words)

    def get_syllable_structure(self, text: str) -> List[str]:
        """Convert text to syllable structure pattern"""
        pattern = ''
        for char in text:
            if char in self.mongolian_vowels:
                pattern += 'V'
            elif char in self.mongolian_consonants:
                pattern += 'C'
            elif char in self.mongolian_modifiers:
                pattern += 'M'
        
        # Split into syllables based on rules
        syllables = []
        current = ''
        
        for i, char in enumerate(pattern):
            current += char
            
            # Check if current pattern forms valid syllable
            if len(current) > 0:
                for valid in self.valid_syllables:
                    if current == valid:
                        syllables.append(current)
                        current = ''
                        break
            
            # Handle remaining characters
            if i == len(pattern) - 1 and current:
                syllables.append(current)
        
        return syllables

    def add_position_aware_encoding(self, text: str) -> List[Tuple[str, torch.Tensor]]:
        # First create list of special tokens for checking
        special_tokens = [
            self.LONG_PAUSE, self.SHORT_PAUSE,
            self.EMPHASIS_START, self.EMPHASIS_END,
            self.QUESTION_START, self.QUESTION_END,
            self.EXCLAMATION_START, self.EXCLAMATION_END,
            '<letter_mn>', '</letter_mn>',
            '<letter_en>', '</letter_en>',
            '<english>', '</english>'
        ]
        
        # Create a pattern that matches any special token
        special_token_pattern = '|'.join(map(re.escape, special_tokens))
        
        # Split text into tokens (special tokens and regular characters)
        tokens = []
        current_pos = 0
        for match in re.finditer(f'({special_token_pattern})|(.)', text):
            if match.group(1):  # Special token
                tokens.append((match.group(1), 'special'))
            else:  # Regular character
                tokens.append((match.group(2), 'char'))
        
        # Now process tokens with correct position awareness
        result = []
        char_count = sum(1 for token, type_ in tokens if type_ == 'char')
        
        for i, (token, type_) in enumerate(tokens):
            if type_ == 'special':
                result.append((token, self.syllable_position_embeddings['isolated']))
            else:
                # Regular characters get position-based embedding
                if char_count == 1:
                    position = 'isolated'
                else:
                    # Safe boundary checks
                    has_prev_special = (i > 0 and tokens[i-1][1] == 'special')
                    has_next_special = (i < len(tokens)-1 and tokens[i+1][1] == 'special')
                    is_first = (i == 0)
                    is_last = (i == len(tokens)-1)
                    
                    if is_first or has_prev_special:
                        position = 'initial'
                    elif is_last or has_next_special:
                        position = 'final'
                    else:
                        position = 'medial'
                
                result.append((token, self.syllable_position_embeddings[position]))
        
        return result

    def handle_abbreviations(self, text: str) -> str:
        def expand_abbreviation(match):
            abbr = match.group(0).replace('.', '')  # Remove any trailing period
            spelled_out = []
            
            for char in abbr:
                if char in self.mongolian_vowels or char in self.mongolian_consonants:
                    # Mongolian letter
                    spelled_out.append(f"<letter_mn>{char}</letter_mn>")
                else:
                    # English letter
                    spelled_out.append(f"<letter_en>{char}</letter_en>")
            
            # Add short pauses between letters
            return f"{self.SHORT_PAUSE} " + f" {self.SHORT_PAUSE} ".join(spelled_out) + f" {self.SHORT_PAUSE}"

        return self.abbreviation_pattern.sub(expand_abbreviation, text)

    def handle_punctuation_and_pauses(self, text: str) -> str:
        # Handle dashes (long pauses)
        text = self.dash_pattern.sub(f' {self.LONG_PAUSE} ', text)
        
        # Handle quoted text (with emphasis)
        def add_emphasis(match):
            quoted_text = match.group(1)
            return f' {self.SHORT_PAUSE} {self.EMPHASIS_START}{quoted_text}{self.EMPHASIS_END} {self.SHORT_PAUSE} '
        
        text = self.quote_pattern.sub(add_emphasis, text)
        
        # Handle exclamation and question marks
        def handle_special_punct(match):
            punct = match.group(0)
            if '!' in punct:
                return f' {self.EXCLAMATION_START}{punct}{self.EXCLAMATION_END} '
            else:
                return f' {self.QUESTION_START}{punct}{self.QUESTION_END} '
                
        text = self.special_punct_pattern.sub(handle_special_punct, text)
        
        # Handle regular punctuation
        text = text.replace(',', f' {self.SHORT_PAUSE} ')
        text = text.replace(';', f' {self.LONG_PAUSE} ')
        text = text.replace(':', f' {self.LONG_PAUSE} ')
        
        return text

    def _get_special_token_weight(self, token: str) -> float:
        """Get weight for special tokens with proper mapping"""
        weights = {
            self.LONG_PAUSE: 2.0,
            self.SHORT_PAUSE: 1.5,
            self.EMPHASIS_START: 1.0,
            self.EMPHASIS_END: 1.0,
            self.QUESTION_START: 1.3,
            self.QUESTION_END: 1.5,
            self.EXCLAMATION_START: 1.4,
            self.EXCLAMATION_END: 1.6,
            '<letter_mn>': 1.2,
            '</letter_mn>': 1.0,
            '<letter_en>': 1.3,
            '</letter_en>': 1.0,
            '<english>': 1.2,
            '</english>': 1.0
        }
        return weights.get(token, 1.0)  # Default weight of 1.0 if token not found

    def get_phoneme_weights(self, text: str) -> torch.Tensor:
        """Calculate phoneme weights with proper token alignment"""
        tokens = self._tokenize_text(text)
        weights = torch.ones(len(tokens))
        
        for i, token in enumerate(tokens):
            # Handle special tokens
            if token in self.special_tokens:
                weights[i] = self._get_special_token_weight(token)
                continue
            
            # Handle regular characters and combinations
            if i < len(tokens) - 1:
                bigram = token + tokens[i+1]
                if bigram in self.phoneme_weights:
                    weights[i] = self.phoneme_weights[bigram]
                    weights[i+1] = self.phoneme_weights[bigram]
                    continue
            
            # Handle single characters
            weights[i] = 1.0
        
        return weights

    def _get_prosody_features(self, text: str) -> Dict[str, torch.Tensor]:
        """Extract prosody features with proper token alignment"""
        # First, tokenize the text
        tokens = self._tokenize_text(text)
        token_count = len(tokens)
        
        features = {
            'emphasis': torch.zeros(token_count),
            'pause_duration': torch.zeros(token_count),
            'intonation': torch.zeros(token_count),
        }
        
        current_idx = 0
        in_emphasis = False
        in_question = False
        in_exclamation = False
        
        for i, token in enumerate(tokens):
            # Handle emphasis
            if token == self.EMPHASIS_START:
                in_emphasis = True
            elif token == self.EMPHASIS_END:
                in_emphasis = False
            elif in_emphasis:
                features['emphasis'][i] = 1.0
                
            # Handle question
            if token == self.QUESTION_START:
                in_question = True
            elif token == self.QUESTION_END:
                in_question = False
            elif in_question:
                features['intonation'][i] = 1.0
                
            # Handle exclamation
            if token == self.EXCLAMATION_START:
                in_exclamation = True
            elif token == self.EXCLAMATION_END:
                in_exclamation = False
            elif in_exclamation:
                features['intonation'][i] = 2.0
                
            # Handle pauses
            if token == self.LONG_PAUSE:
                features['pause_duration'][i] = 2.0
            elif token == self.SHORT_PAUSE:
                features['pause_duration'][i] = 1.0
                
        return features

    def _tokenize_text(self, text: str) -> List[str]:
        """Split text into tokens while preserving special tokens"""
        special_tokens = set([
            self.LONG_PAUSE, self.SHORT_PAUSE,
            self.EMPHASIS_START, self.EMPHASIS_END,
            self.QUESTION_START, self.QUESTION_END,
            self.EXCLAMATION_START, self.EXCLAMATION_END,
            '<letter_mn>', '</letter_mn>',
            '<letter_en>', '</letter_en>',
            '<english>', '</english>'
        ])
        
        # Create pattern for tokenization
        pattern = '|'.join(map(re.escape, special_tokens)) + '|.'
        
        return [match.group(0) for match in re.finditer(pattern, text)]

    def process_text(self, text: str) -> Tuple[List[Tuple[str, torch.Tensor]], torch.Tensor, Dict[str, torch.Tensor]]:
        # 1. Handle abbreviations first
        text = self.handle_abbreviations(text)
        
        # 2. Handle punctuation and pauses
        text = self.handle_punctuation_and_pauses(text)
        
        # 3. Handle numbers and Roman numerals
        text = self.convert_numbers(text)
        
        # 4. Apply vowel harmony
        text = self.apply_vowel_harmony(text)
        
        # 5. Handle English words
        text = self.handle_english_words(text)
        
        # 6. Add position-aware encoding
        tokens_with_position = self.add_position_aware_encoding(text)
        
        # 7. Calculate phoneme weights
        weights = self.get_phoneme_weights(text)
        
        # 8. Extract prosody features
        prosody_features = self._get_prosody_features(text)
        
        return tokens_with_position, weights, prosody_features
    
    
class MongolianNumberConverter:
    def __init__(self):
        self.ones = {
            0: '', 1: 'нэг', 2: 'хоёр', 3: 'гурав', 4: 'дөрөв',
            5: 'тав', 6: 'зургаа', 7: 'долоо', 8: 'найм', 9: 'ес'
        }
        
        self.tens = {
            2: 'хорь', 3: 'гуч', 4: 'дөч', 5: 'тавь',
            6: 'жар', 7: 'дал', 8: 'ная', 9: 'ер'
        }
        
        self.powers = {
            3: 'мянга',
            6: 'сая',
            9: 'тэрбум',
            12: 'их наяд'
        }
        
        self.roman_numerals = {
            'M': 1000, 'CM': 900, 'D': 500, 'CD': 400,
            'C': 100, 'XC': 90, 'L': 50, 'XL': 40,
            'X': 10, 'IX': 9, 'V': 5, 'IV': 4, 'I': 1
        }

    def convert_to_words(self, number: int) -> str:
        if number == 0:
            return 'тэг'
            
        words = []
        
        # Handle negative numbers
        if number < 0:
            words.append('сөрөг')
            number = abs(number)
        
        # Convert each group of three digits
        def convert_group(n: int) -> str:
            if n == 0:
                return ''
                
            result = []
            
            # Handle hundreds
            hundreds = n // 100
            if hundreds > 0:
                result.append(f"{self.ones[hundreds]} зуун")
            
            # Handle tens and ones
            remainder = n % 100
            if remainder > 0:
                if remainder < 10:
                    result.append(self.ones[remainder])
                else:
                    tens_digit = remainder // 10
                    ones_digit = remainder % 10
                    
                    if tens_digit >= 2:
                        result.append(self.tens[tens_digit])
                    if ones_digit > 0:
                        result.append(self.ones[ones_digit])
            
            return ' '.join(result)
        
        # Process number in groups of three digits
        power = 0
        while number > 0:
            group = number % 1000
            if group > 0:
                group_text = convert_group(group)
                if power > 0 and power in self.powers:
                    group_text += f" {self.powers[power]}"
                words.insert(0, group_text)
            number //= 1000
            power += 3
        
        return ' '.join(words)

    def convert_roman_to_mongolian(self, roman: str) -> str:
        # Convert Roman numeral to integer
        number = 0
        prev_value = 0
        
        for char in reversed(roman.upper()):
            current_value = self.roman_numerals[char]
            if current_value >= prev_value:
                number += current_value
            else:
                number -= current_value
            prev_value = current_value
        
        # Convert integer to Mongolian words
        return self.convert_to_words(number)