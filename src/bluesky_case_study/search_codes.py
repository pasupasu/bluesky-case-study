# Tier 1: unambiguous terms. The 1F... codes are 🇵🇸, 🇮🇱 and 🍉 which don't play nice with word boundaries \b
CORE = (
    r"(?i)\b(?:"
    r"israel\w*|palestin\w*|gazan?s?|west\s+bank|hamas|idf|zionis\w*|"
    r"netanyahu|intifada|nakba|rafah|khan\s+y[ou]nis|al[-\s]?aqsa|"
    r"tel\s+aviv|from\s+the\s+river\s+to\s+the\s+sea"
    r")\b"
    r"|\x{1F1F5}\x{1F1F8}|\x{1F1EE}\x{1F1F1}|\x{1F349}"
)

# Tier 2: often about I/P in this period, but not always.
CONTEXT = (
    r"(?i)\b(?:"
    r"ceasefire|cease[-\s]fire|genocide|oct(?:ober)?\.?\s*7(?:th)?|"
    r"antisemit\w*|hezbollah|jerusalem|icj|hostages?|two[-\s]state"
    r")\b"
)