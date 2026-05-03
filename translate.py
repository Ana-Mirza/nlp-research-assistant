# Multi-language translation module using facebook/nllb-200-distilled-600M.
#
# Why NLLB-200: Meta's No Language Left Behind model supports 200+ languages
# in a single compact model. The distilled-600M variant offers a good balance
# between translation quality and resource usage, making it practical for
# CPU-only inference in research pipelines.

from langdetect import detect, LangDetectException
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Maps ISO 639-1 codes (from langdetect) to NLLB BCP-47-style language codes.
NLLB_LANG_MAP = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn", "pl": "pol_Latn",
    "ru": "rus_Cyrl", "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "ar": "arb_Arab", "hi": "hin_Deva", "tr": "tur_Latn", "vi": "vie_Latn",
    "th": "tha_Thai", "sv": "swe_Latn", "da": "dan_Latn", "fi": "fin_Latn",
    "no": "nob_Latn", "cs": "ces_Latn", "ro": "ron_Latn", "hu": "hun_Latn",
    "uk": "ukr_Cyrl", "ca": "cat_Latn", "hr": "hrv_Latn", "bg": "bul_Cyrl",
    "sk": "slk_Latn", "el": "ell_Grek", "he": "heb_Hebr", "id": "ind_Latn",
    "ms": "zsm_Latn", "bn": "ben_Beng", "ta": "tam_Taml", "te": "tel_Telu",
}

# Lazy-loaded globals
_tokenizer = None
_model = None


def _load_model():
    """Load model and tokenizer on first use."""
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def _translate(text: str, src_nllb: str, tgt_nllb: str) -> str:
    """Run NLLB translation from src language to target language."""
    _load_model()
    _tokenizer.src_lang = src_nllb
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    tgt_id = _tokenizer.convert_tokens_to_ids(tgt_nllb)
    outputs = _model.generate(**inputs, forced_bos_token_id=tgt_id, max_new_tokens=512)
    return _tokenizer.decode(outputs[0], skip_special_tokens=True)


def detect_language(text: str) -> str:
    """Detect the language of the input text.

    Args:
        text: Input text to detect.

    Returns:
        ISO 639-1 language code (e.g., 'en', 'es', 'fr').
        Falls back to 'en' if detection fails or text is empty.
    """
    if not text or not text.strip():
        return "en"
    try:
        return detect(text)
    except LangDetectException:
        return "en"


def translate_to_english(text: str) -> tuple[str, str]:
    """Translate text from its detected language to English.

    Args:
        text: Input text in any supported language.

    Returns:
        Tuple of (translated_text, detected_lang_code).
        If text is already English or empty, returns (text, 'en').
        If the detected language is unsupported, returns (text, detected_code).
    """
    if not text or not text.strip():
        return (text, "en")

    lang = detect_language(text)
    if lang == "en":
        return (text, "en")

    src_nllb = NLLB_LANG_MAP.get(lang)
    if src_nllb is None:
        return (text, lang)

    return (_translate(text, src_nllb, "eng_Latn"), lang)


def translate_from_english(text: str, target_lang: str) -> str:
    """Translate English text to the specified target language.

    Args:
        text: English input text.
        target_lang: ISO 639-1 code of the target language (e.g., 'es', 'fr').

    Returns:
        Translated text, or the original text if target is 'en',
        the language is unsupported, or the text is empty.
    """
    if not text or not text.strip() or target_lang == "en":
        return text

    tgt_nllb = NLLB_LANG_MAP.get(target_lang)
    if tgt_nllb is None:
        return text

    return _translate(text, "eng_Latn", tgt_nllb)
