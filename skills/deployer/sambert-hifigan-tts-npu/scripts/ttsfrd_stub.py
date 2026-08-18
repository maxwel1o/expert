#!/usr/bin/env python3
"""
ttsfrd stub module - pure Python replacement for the closed-source ttsfrd library.
Provides text-to-symbol conversion for Sambert-Hifigan TTS models using pypinyin.
"""

import os
import re
import warnings


TONE_MAP = {
    '1': 'tone1', '2': 'tone2', '3': 'tone3',
    '4': 'tone4', '5': 'tone5', '0': 'tone0',
    '': 'tone_none',
}

# Some phone sets use different symbols for certain pinyin initials
PINYIN_INITIAL_MAP = {
    'x': 'xx',
}


class TtsFrontendEngine:
    """Mock TtsFrontendEngine that uses pure Python for Chinese text processing."""

    def __init__(self):
        self._initialized = False
        self._lang_type = None
        self._resources_dir = None
        self._sy_dict = {}
        self._tone_dict = {}
        self._speaker_dict = {}
        self._emo_dict = {}
        self._syllable_dict = {}
        self._word_dict = {}

    def initialize(self, resource_path):
        """Initialize with resource directory."""
        self._resources_dir = resource_path
        self._initialized = True
        return True

    def set_lang_type(self, lang_type):
        """Set language type."""
        self._lang_type = lang_type
        return True

    def _pinyin_to_sy(self, pinyin, tone):
        """Convert pinyin to model syllable format."""
        # Remove the tone number from pinyin
        base = re.sub(r'[0-5]$', '', pinyin)
        return base + '_c'

    def _text_to_symbols_pinyin(self, text, speaker="F7"):
        """Convert Chinese text to pinyin-based symbols."""
        try:
            from pypinyin import pinyin as pypinyin_func, Style
        except ImportError:
            raise ImportError("pypinyin is required. Install with: pip install pypinyin")

        # Keep only CJK Unified Ideographs
        cjk_chars = re.findall(r'[一-鿿]', text)
        if not cjk_chars:
            return ''

        # Get pinyin with tones
        py_result = pypinyin_func(cjk_chars, style=Style.TONE3)

        symbols = []
        for i, (p,) in enumerate(py_result):
            # Extract base syllable and tone
            match = re.match(r'([a-z]+)([0-5])?', p.lower())
            if match:
                base = match.group(1)
                tone = match.group(2) or '5'

                sy = base + '_c'
                # Apply pinyin-to-phone mapping
                if base in PINYIN_INITIAL_MAP:
                    sy = PINYIN_INITIAL_MAP[base] + '_c'
                tone_str = f'tone{tone}'

                # Determine syllable flag
                if len(py_result) == 1:
                    s_flag = 's_both'
                elif i == 0:
                    s_flag = 's_begin'
                elif i == len(py_result) - 1:
                    s_flag = 's_end'
                else:
                    s_flag = 's_middle'

                # Determine word segment (simplified: each character is its own word)
                if len(py_result) == 1:
                    w_flag = 'word_both'
                elif i == 0:
                    w_flag = 'word_begin'
                elif i == len(py_result) - 1:
                    w_flag = 'word_end'
                else:
                    w_flag = 'word_middle'

                emotion = 'emotion_neutral'

                symbol = f"{{{sy}${tone_str}${s_flag}${w_flag}${emotion}${speaker}}}"
                symbols.append(symbol)

        return ' '.join(symbols)

    def _english_to_symbols(self, text, speaker="F7"):
        """Convert English text to symbols."""
        words = text.lower().split()
        symbols = []
        for i, word in enumerate(words):
            # Use @en symbol for English words
            sy = '@en'
            tone = 'tone_none'

            if len(words) == 1:
                s_flag = 's_both'
            elif i == 0:
                s_flag = 's_begin'
            elif i == len(words) - 1:
                s_flag = 's_end'
            else:
                s_flag = 's_middle'

            if len(words) == 1:
                w_flag = 'word_both'
            elif i == 0:
                w_flag = 'word_begin'
            elif i == len(words) - 1:
                w_flag = 'word_end'
            else:
                w_flag = 'word_middle'

            emotion = 'emotion_neutral'
            symbol = f"{{{sy}${tone}${s_flag}${w_flag}${emotion}${speaker}}}"
            symbols.append(symbol)

        return ' '.join(symbols)

    def _english_to_phonemes(self, text):
        """Convert English text to ARPAbet phonemes using g2p."""
        try:
            from g2p import make_g2p
            g2p = make_g2p('eng', 'eng-arpabet')
        except ImportError:
            raise ImportError("g2p is required for English TTS. Install with: pip install g2p")

        words = re.findall(r"[a-zA-Z']+", text)
        symbols = []

        for wi, word in enumerate(words):
            phones = str(g2p(word)).strip().split()
            phones = [p.lower() for p in phones if p]
            if not phones:
                continue
            n_phones = len(phones)
            is_first_word = (wi == 0)
            is_last_word = (wi == len(words) - 1)
            for j, phone in enumerate(phones):
                if n_phones == 1:
                    s_flag = 's_both'
                elif j == 0:
                    s_flag = 's_begin'
                elif j == n_phones - 1:
                    s_flag = 's_end'
                else:
                    s_flag = 's_middle'
                if is_first_word and is_last_word:
                    w_flag = 'word_both'
                elif is_first_word:
                    w_flag = 'word_begin'
                elif is_last_word:
                    w_flag = 'word_end'
                else:
                    w_flag = 'word_middle'
                symbol = f"{{{phone}$tone_none${s_flag}${w_flag}$emotion_neutral$F7}}"
                symbols.append(symbol)
        return ' '.join(symbols)

    def gen_tacotron_symbols(self, text):
        """Convert text to tacotron symbols (mimics ttsfrd behavior)."""
        text = text.strip().replace('\n', ' ')
        has_chinese = bool(re.search(r'[一-鿿]', text))
        if has_chinese:
            symbol_str = self._text_to_symbols_pinyin(text)
        else:
            symbol_str = self._english_to_phonemes(text)
        return f"0_0\t{symbol_str}"


def text_to_mit_symbols(texts, resources_dir, speaker, lang="PinYin"):
    """Mock text_to_mit_symbols function."""
    engine = TtsFrontendEngine()
    engine.initialize(resources_dir)
    engine.set_lang_type(lang)

    symbols_lst = []
    for idx, text in enumerate(texts):
        text = text.strip()
        res = engine.gen_tacotron_symbols(text)
        res = res.replace("F7", speaker)
        sentences = res.split("\n")
        for sentence in sentences:
            arr = sentence.split("\t")
            if len(arr) != 2:
                continue
            sub_index, symbols = sentence.split("\t")
            symbol_str = "{}_{}\t{}\n".format(idx, sub_index, symbols)
            symbols_lst.append(symbol_str)

    return symbols_lst


if __name__ == "__main__":
    engine = TtsFrontendEngine()
    engine.initialize("/tmp")
    result = engine.gen_tacotron_symbols("北京今天天气怎么样")
    print(result)
