# Multi-language translation module using facebook/nllb-200-distilled-600M.
#
# Why NLLB-200: Meta's No Language Left Behind model supports 200+ languages
# in a single compact model. The distilled-600M variant offers a good balance
# between translation quality and resource usage, making it practical for
# CPU-only inference in research pipelines.

from langid.langid import LanguageIdentifier, model as langid_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import LANG_DETECT_MIN_CONFIDENCE

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Maps ISO 639-1 codes (from langid) to NLLB BCP-47-style language codes.
NLLB_LANG_MAP = {
    "af": "afr_Latn", "ar": "arb_Arab", "bg": "bul_Cyrl", "bn": "ben_Beng",
    "ca": "cat_Latn", "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn",
    "de": "deu_Latn", "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn",
    "et": "est_Latn", "fa": "pes_Arab", "fi": "fin_Latn", "fr": "fra_Latn",
    "gu": "guj_Gujr", "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn",
    "hu": "hun_Latn", "id": "ind_Latn", "it": "ita_Latn", "ja": "jpn_Jpan",
    "kn": "kan_Knda", "ko": "kor_Hang", "lt": "lit_Latn", "lv": "lvs_Latn",
    "mk": "mkd_Cyrl", "ml": "mal_Mlym", "mr": "mar_Deva", "ne": "npi_Deva",
    "nl": "nld_Latn", "no": "nob_Latn", "pa": "pan_Guru", "pl": "pol_Latn",
    "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "sk": "slk_Latn",
    "sl": "slv_Latn", "so": "som_Latn", "sq": "als_Latn", "sv": "swe_Latn",
    "sw": "swh_Latn", "ta": "tam_Taml", "te": "tel_Telu", "th": "tha_Thai",
    "tl": "tgl_Latn", "tr": "tur_Latn", "uk": "ukr_Cyrl", "ur": "urd_Arab",
    "vi": "vie_Latn", "zh-cn": "zho_Hans", "zh-tw": "zho_Hant",
}

# Lazy-loaded globals
_tokenizer = None
_model = None
_lang_identifier = None


def _get_lang_identifier():
    """Lazy-load a langid identifier configured to return normalized probabilities in [0, 1]."""
    global _lang_identifier
    if _lang_identifier is None:
        _lang_identifier = LanguageIdentifier.from_modelstring(langid_model, norm_probs=True)
    return _lang_identifier


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
    """Detect the language of the input text using langid.

    Uses normalized probabilities in [0, 1]. Falls back to 'en' when the
    detector's confidence is below LANG_DETECT_MIN_CONFIDENCE — short or
    ambiguous queries can misclassify, and defaulting to English is the
    safest behavior since the corpus is English.

    Returns:
        ISO 639-1 language code (e.g., 'en', 'es', 'fr').
        Falls back to 'en' if text is empty, detection fails, or confidence
        is below the configured threshold.
    """
    if not text or not text.strip():
        return "en"
    try:
        identifier = _get_lang_identifier()
        lang, prob = identifier.classify(text)
        if prob < LANG_DETECT_MIN_CONFIDENCE:
            return "en"
        # langid struggles with short Romance language texts (ro/fr/it/es confusion).
        # Fall back to langdetect for short texts or ambiguous Romance languages.
        ambiguous_romance = {"fr", "it", "ro", "es", "pt", "ca"}
        if len(text.split()) < 10 and lang in ambiguous_romance:
            try:
                from langdetect import detect_langs
                candidates = detect_langs(text)
                for c in candidates:
                    if c.lang in NLLB_LANG_MAP:
                        return c.lang
            except Exception:
                pass
        return lang
    except Exception:
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
