VALID_90_CLASSES = {
    'a', 'ba', 'be', 'bi', 'bo', 'bu', 'ca', 'ce', 'ci', 'co', 'cu', 
    'da', 'de', 'di', 'do', 'du', 'e', 'fa', 'fe', 'fi', 'fo', 'fu', 
    'ga', 'ge', 'gi', 'go', 'gu', 'ha', 'he', 'hi', 'ho', 'hu', 'i', 
    'ja', 'je', 'ji', 'jo', 'ju', 'ka', 'ke', 'ki', 'ko', 'ku', 'la', 
    'le', 'li', 'lo', 'lu', 'ma', 'me', 'mi', 'mo', 'mu', 'na', 'ne', 
    'ni', 'no', 'nu', 'o', 'pa', 'pe', 'pi', 'po', 'pu', 'ra', 're', 
    'ri', 'ro', 'ru', 'sa', 'se', 'si', 'so', 'su', 'ta', 'te', 'ti', 
    'to', 'tu', 'u', 'wa', 'we', 'wi', 'wo', 'wu', 'ya', 'ye', 'yi', 
    'yo', 'yu'
}

def bima_to_syllables(word):
    word = word.lower().replace("'", "").replace("-", "").strip()
    syllables = []
    i = 0
    n = len(word)
    
    while i < n:
        if i + 3 <= n and word[i:i+3] == "ngg":
            if i + 4 <= n and word[i+3] in "aiueo":
                vowel = word[i+3]
                syllable_cv = f"g{vowel}"
                if syllable_cv in VALID_90_CLASSES:
                    syllables.append(syllable_cv)
                    i += 4
                    continue

        if i + 3 <= n and word[i:i+2] in ["mb", "nd", "mp", "nt", "nc", "nk"]:
            konsonan_dasar = word[i+1]
            vowel = word[i+2]
            syllable_cv = f"{konsonan_dasar}{vowel}"
            if syllable_cv in VALID_90_CLASSES:
                syllables.append(syllable_cv)
                i += 3
                continue

        if i + 1 < n and word[i:i+2] in VALID_90_CLASSES:
            syllables.append(word[i:i+2])
            i += 2
        elif word[i] in VALID_90_CLASSES:
            syllables.append(word[i])
            i += 1
        else:
            i += 1
            
    return syllables

if __name__ == "__main__":
    test_words = ["dou", "mbojo", "kandei", "nggahi", "ina", "ama", "surat"]
    for w in test_words:
        print(f"Kata: {w:<10} -> Suku Kata: {bima_to_syllables(w)}")
