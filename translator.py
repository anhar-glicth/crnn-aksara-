import re

BIMA_LEXICON = {
    "saya": "nahu", "aku": "nahu", "kamu": "ita", "engkau": "ita",
    "dia": "sia", "ia": "sia", "kami": "mami", "kita": "mami",
    "mereka": "siadahu", "kalian": "itadahu",
    "orang": "dou", "manusia": "dou", "ayah": "ama", "bapak": "ama",
    "ibu": "ina", "anak": "anada", "saudara": "wari", "kakak": "kaka",
    "adik": "wari", "raja": "ruma", "tuan": "ruma", "teman": "soa",
    "makan": "ngaha", "minum": "nono", "tidur": "tente", "bangun": "tente",
    "pergi": "lao", "jalan": "lampa", "datang": "mai", "pulang": "bali",
    "lihat": "nonto", "bicara": "nggahi", "kata": "nggahi", "bahasa": "nggahi",
    "tulis": "buri", "baca": "baca", "ambil": "alha", "bawa": "hanta",
    "duduk": "kadodo", "mandi": "doro", "beli": "weli", "jual": "weli",
    "kerja": "karawi", "bantu": "tolo", "ingat": "karingi", "tahu": "bade",
    "air": "oi", "api": "afi", "tanah": "dana", "gunung": "doro",
    "negeri": "dana", "kampung": "rasa", "desa": "rasa", "rumah": "uma",
    "surat": "surat", "batu": "wadu", "pohon": "pu'u", "daun": "ro'o",
    "laut": "motito", "sungai": "keli", "hutan": "doro", "matahari": "mata lolu",
    "bulan": "wura", "bintang": "sari", "angin": "angi",
    "baik": "taho", "bagus": "taho", "buruk": "kacumpu", "rusak": "kacumpu",
    "besar": "na'e", "kecil": "to'i", "panjang": "kula", "pendek": "bondo",
    "panas": "pana", "dingin": "morina", "cantik": "ganta", "tampan": "malingi",
    "senang": "taho ruku", "sedih": "susah", "benar": "taho", "salah": "sala",
    "baru": "bou", "lama": "toho",
    "satu": "ica", "dua": "dua", "tiga": "tolu", "empat": "upa", "lima": "lima",
    "enam": "ina", "tujuh": "pidu", "delapan": "walu", "sembilan": "ciwi", "sepuluh": "sampulu",
    "dan": "labo", "dengan": "labo", "ke": "di", "di": "di", "dari": "mai",
    "apa": "au", "siapa": "co'o", "mana": "be", "bagaimana": "bune", "kapan": "piri",
    "tidak": "wara", "bukan": "wara", "ada": "wara", "bima": "mbojo"
}

def translate_indonesia_to_bima(text):
    clean_text = text.lower().strip()
    words = re.findall(r'\b[\w\'-]+\b', clean_text)
    
    translated_tokens = []
    details = []
    
    for w in words:
        if w in BIMA_LEXICON:
            bima_word = BIMA_LEXICON[w]
            translated_tokens.append(bima_word)
            details.append({"source": w, "target": bima_word, "is_known": True})
        else:
            translated_tokens.append(w)
            details.append({"source": w, "target": w, "is_known": False})
            
    return " ".join(translated_tokens), details

if __name__ == "__main__":
    contoh = "orang bima berbicara bahasa tanah air"
    hasil, det = translate_indonesia_to_bima(contoh)
    print("Input :", contoh)
    print("Output:", hasil)
    print("Detail:", det)
