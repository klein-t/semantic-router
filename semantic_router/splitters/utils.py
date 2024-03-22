import regex
import tiktoken
from semantic_router.utils.logger import logger

def split_to_sentences(text: str, regex_pattern: str) -> list[str]:
    """
    Enhanced regex pattern to split a given text into sentences more accurately.

    The enhanced regex pattern includes handling for:
    - Direct speech and quotations.
    - Abbreviations, initials, and acronyms.
    - Decimal numbers and dates.
    - Ellipses and other punctuation marks used in informal text.
    - Removing control characters and format characters.

    Args:
        text (str): The text to split into sentences.

    Returns:
        list: A list of sentences extracted from the text.
    """
    if regex_pattern is None: 
        regex_pattern = r"""
            # Negative lookbehind for word boundary, word char, dot, word char
            (?<!\b\w\.\w.)
            # Negative lookbehind for single uppercase initials like "A."
            (?<!\b[A-Z][a-z]\.)
            # Negative lookbehind for abbreviations like "U.S."
            (?<!\b[A-Z]\.)
            # Negative lookbehind for abbreviations with uppercase letters and dots
            (?<!\b\p{Lu}\.\p{Lu}.)
            # Negative lookbehind for numbers, to avoid splitting decimals
            (?<!\b\p{N}\.)
            # Positive lookbehind for punctuation followed by whitespace
            (?<=\.|\?|!|:|\.\.\.)\s+
            # Positive lookahead for uppercase letter or opening quote at word boundary
            (?="?(?=[A-Z])|"\b)
            # OR
            |
            # Splits after punctuation that follows closing punctuation, followed by
            # whitespace
            (?<=[\"\'\]\)\}][\.!?])\s+(?=[\"\'\(A-Z])
            # OR
            |
            # Splits after punctuation if not preceded by a period
            (?<=[^\.][\.!?])\s+(?=[A-Z])
            # OR
            |
            # Handles splitting after ellipses
            (?<=\.\.\.)\s+(?=[A-Z])
            # OR
            |
            # Matches and removes control characters and format characters
            [\p{Cc}\p{Cf}]+
        """
    elif not isinstance(regex_pattern, str):
        raise TypeError("regex_pattern must be a string.")
    sentences = regex.split(regex_pattern, text, flags=regex.VERBOSE)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


def split_to_sentences_spacy(text: str, spacy_model: str = "en_core_web_sm") -> list[str]:
    """
    Use SpaCy to split a given text into sentences. Supported languages: English.

    Args:
        text (str): The text to split into sentences.

    Returns:
        list: A list of sentences extracted from the text.
    """

    # Check if SpaCy is installed
    try:
        import spacy
    except ImportError:
        logger.warning(
            "SpaCy is not installed. Please `pip install "
            "semantic-router[processing]`."
        )
        return

    # Check if the SpaCy model is installed
    try:
        spacy.load(spacy_model)
    except OSError:
        print(f"Spacy model '{spacy_model}' not found, downloading...")
        from spacy.cli import download
        download(spacy_model)
        print(f"Downloaded and installed model '{spacy_model}'.")

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sentence.text.strip() for sentence in doc.sents]
    return sentences

def split_to_sentences_recursive(text: str, regex_separators: str):
    """
    Split a given text into sentences using a recursive text splitter from LangChain.

    Args:
        text (str): The text to split into sentences.

    Returns:
        list: A list of sentences extracted from the text.
    """

    # Check if langchain_text_splitters are installed
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        logger.warning(
            "langchain_text_splitters is not installed. Please `pip install "
            "semantic-router[processing]`."
        )
        return
    
    if regex_separators is None:
        regex_separators = ["\n", "\n\n", r'(?<=\.)\s+']
    elif not isinstance(regex_separators, str):
        raise TypeError("regex_separators must be a string.")
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=30,
    chunk_overlap=0,
    length_function=tiktoken_length,
    separators=regex_separators,
    is_separator_regex=True,
    )

    texts = text_splitter.create_documents([text])
    results = [text.page_content for text in texts]
    return results

def tiktoken_length(text: str) -> int:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)