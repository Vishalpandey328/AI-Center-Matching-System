import re

# Dictionary for common word replacements
REPLACEMENTS = {
    "govt": "government",
    "govt.": "government",
    "mahila": "girls",
    "sr sec": "senior secondary",
    "sr. sec.": "senior secondary",
    "jr": "junior",
    "inst": "institute",
    "sch": "school"
}


def clean_text(text):
    """
    Clean and normalize center names
    """

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Replace known variations
    for key, value in REPLACEMENTS.items():
        text = text.replace(key, value)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()