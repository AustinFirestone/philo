import sqlite3
import datetime
from collections import Counter
import sys
import os

try:
    import spacy
    from spacy.cli import download as spacy_download
    from deep_translator import GoogleTranslator
except ImportError as exc:
    print("Error: Missing dependencies.")
    print("Run: pip install spacy deep-translator")
    sys.exit(1)

class LinguisticEngine:
    def __init__(self, db_name="linguist.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self.nlp = self._ensure_russian_model()
        self.translator = GoogleTranslator(source='ru', target='en')
        self.setup_database()

    def _ensure_russian_model(self, model_name="ru_core_news_lg"):
        print(f"Loading NLP Model ({model_name})...")
        try:
            return spacy.load(model_name)
        except OSError:
            print(f"Downloading '{model_name}'...")
            spacy_download(model_name)
            return spacy.load(model_name)

    def setup_database(self):
        self.cursor.executescript('''
            CREATE TABLE IF NOT EXISTS global_frequency (
                lemma TEXT PRIMARY KEY,
                rank INTEGER
            );
            CREATE TABLE IF NOT EXISTS known_vocabulary (
                lemma TEXT PRIMARY KEY,
                added_date DATE
            );
            CREATE TABLE IF NOT EXISTS flashcards (
                lemma TEXT PRIMARY KEY,
                translation TEXT,
                interval INTEGER,
                repetitions INTEGER,
                ease_factor REAL,
                next_review DATE,
                last_reviewed DATE
            );
        ''')
        # Migrations: Silently add new columns if upgrading from older db versions
        try: self.cursor.execute("ALTER TABLE flashcards ADD COLUMN translation TEXT")
        except sqlite3.OperationalError: pass
            
        try: self.cursor.execute("ALTER TABLE flashcards ADD COLUMN last_reviewed DATE")
        except sqlite3.OperationalError: pass
            
        self.conn.commit()

    def import_50k_frequency(self, filepath="ru_50k.txt"):
        if not os.path.exists(filepath):
            print(f"\nError: {filepath} not found.")
            return
        print("\nImporting 50,000 word frequency list...")
        self.cursor.execute('DELETE FROM global_frequency') 
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for rank, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) >= 1:
                    data.append((parts[0].lower(), rank))
        self.cursor.executemany('INSERT OR IGNORE INTO global_frequency (lemma, rank) VALUES (?, ?)', data)
        self.conn.commit()
        print(f"Successfully seeded database with {len(data)} words!")

    def load_known_vocab(self, filepath):
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found.")
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        today_str = datetime.date.today().isoformat()
        data = [(word, today_str) for word in words]
        self.cursor.executemany('INSERT OR IGNORE INTO known_vocabulary (lemma, added_date) VALUES (?, ?)', data)
        self.conn.commit()
        print(f"Successfully imported {len(words)} known words.")

    def analyze_text(self, filepath):
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found.")
            return

        print("\nParsing text and extracting contexts...")
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        doc = self.nlp(text)
        
        lemma_counts = Counter()
        lemma_to_sentence = {}
        
        # Extract lemmas and their first contextual sentence
        for sent in doc.sents:
            clean_sent = " ".join(sent.text.split()) 
            for token in sent:
                if token.is_alpha:
                    lemma = token.lemma_.lower()
                    lemma_counts[lemma] += 1
                    if lemma not in lemma_to_sentence:
                        lemma_to_sentence[lemma] = clean_sent
        
        # Filter out already known words
        self.cursor.execute('SELECT lemma FROM known_vocabulary')
        known_set = set(row[0] for row in self.cursor.fetchall())
        unknown_lemmas = {lemma: count for lemma, count in lemma_counts.items() if lemma not in known_set}
        
        q1, q2, q3, q4 = [], [], [], []
        HIGH_BOOK_FREQ = 2 
        HIGH_RU_FREQ_RANK = 15000 

        # Sort into Quadrants
        for lemma, book_count in unknown_lemmas.items():
            self.cursor.execute('SELECT rank FROM global_frequency WHERE lemma = ?', (lemma,))
            result = self.cursor.fetchone()
            global_rank = result[0] if result else 999999
            
            if book_count >= HIGH_BOOK_FREQ and global_rank <= HIGH_RU_FREQ_RANK: q1.append(lemma)
            elif book_count < HIGH_BOOK_FREQ and global_rank <= HIGH_RU_FREQ_RANK: q2.append(lemma)
            elif book_count >= HIGH_BOOK_FREQ and global_rank > HIGH_RU_FREQ_RANK: q3.append(lemma)
            else: q4.append(lemma)

        print(f"\n=== TEXT ANALYTICS REPORT ===")
        print(f"Total words (tokens): {sum(lemma_counts.values())}")
        print(f"Unique words (lemmas): {len(lemma_counts)}")
        print(f"Known words avoided: {len(lemma_counts) - len(unknown_lemmas)}")
        print(f"Target unknown words: {len(unknown_lemmas)}")
        print("\n--- 4-QUADRANT DISTRIBUTION ---")
        print(f"Q1 (High Book / High RU): {len(q1)} words")
        print(f"Q2 (Low Book  / High RU): {len(q2)} words")
        print(f"Q3 (High Book / Low RU):  {len(q3)} words (Hapax/Motifs)")
        print(f"Q4 (Low Book  / Low RU):  {len(q4)} words (Rare/Archaic)")
        print("-------------------------------")

        export_choice = input("\nExport Markdown Cheat Sheet (All Quadrants)? (y/n): ")
        if export_choice.lower() == 'y':
            self.export_report(filepath, sum(lemma_counts.values()), len(lemma_counts), len(known_set), unknown_lemmas, q1, q2, q3, q4)

        if q1 or q2:
            choice = input(f"\nGenerate TTS Script & Anki Deck for {len(q1) + len(q2)} target words (Q1 & Q2)? (y/n): ")
            if choice.lower() == 'y':
                target_words = q1 + q2
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                self.export_tts_and_anki(target_words, lemma_to_sentence, base_name)
                
                db_choice = input("Add these targets to internal terminal flashcard DB? (y/n): ")
                if db_choice.lower() == 'y':
                    self.generate_flashcards(target_words)

    def export_report(self, original_file, tokens, unique, known, unknown_dict, q1, q2, q3, q4):
        report_name = f"cheat_sheet_{os.path.splitext(os.path.basename(original_file))[0]}.md"
        with open(report_name, 'w', encoding='utf-8') as f:
            f.write(f"# Linguistic Target Sheet: {original_file}\n\n")
            f.write("## Global Metrics\n")
            f.write(f"- **Total Tokens:** {tokens}\n")
            f.write(f"- **Unique Lemmas:** {unique}\n")
            f.write(f"- **Known Words Avoided:** {unique - len(unknown_dict)}\n")
            f.write(f"- **Unknown Target Words:** {len(unknown_dict)}\n\n")
            f.write("## Target Vocabulary (Q1 & Q2 - Under 15k Frequency)\n")
            f.write(f"{', '.join(q1 + q2) if (q1 or q2) else 'None'}\n\n")
            f.write("## Cheat Sheet (Q3 & Q4 - Rare/Archaic/Motifs)\n")
            f.write(f"{', '.join(q3 + q4) if (q3 or q4) else 'None'}\n\n")
        print(f"\n[+] Cheat sheet exported to: {report_name}")

    def export_tts_and_anki(self, words, lemma_to_sentence, filename_prefix):
        print(f"Fetching translations... (Connecting to Google Translate)")
        
        tts_lines = []
        anki_lines = []
        
        for word in words:
            try:
                translation = self.translator.translate(word)
            except Exception:
                translation = "Translation failed"
            
            context = lemma_to_sentence.get(word, "")
            
            # TTS Format: word (pause) translation (pause) sentence
            tts_lines.append(f"{word} (pause) {translation} (pause) {context}")
            
            # Anki Format: Front <tab> Back (Translation + Context)
            anki_lines.append(f"{word}\t{translation}<br><br><i>{context}</i>")
            
        tts_filename = f"{filename_prefix}_tts_script.txt"
        with open(tts_filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(tts_lines))
            
        anki_filename = f"{filename_prefix}_anki_deck.txt"
        with open(anki_filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(anki_lines))
            
        print(f"\n[+] SUCCESS! Engine Output:")
        print(f"    -> Audio Script: {tts_filename}")
        print(f"    -> Anki Deck:    {anki_filename} (Import into Anki as 'Text separated by tabs')")

    def generate_flashcards(self, words):
        today_str = datetime.date.today().isoformat()
        data = []
        for word in words:
            try:
                translation = self.translator.translate(word)
            except Exception:
                translation = "Translation failed"
            data.append((word, translation, 0, 0, 2.5, today_str, today_str))
        
        self.cursor.executemany('''
            INSERT OR IGNORE INTO flashcards (lemma, translation, interval, repetitions, ease_factor, next_review, last_reviewed) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data)
        self.conn.commit()
        print(f"Added {len(words)} words to your internal deck.")

    def run_cli(self):
        while True:
            print("\n[ SOVEREIGN LINGUIST CORE ENGINE ]")
            print("1. Analyze Native Text & Generate Pipelines")
            print("2. Seed 50k Frequency Database")
            print("3. Import Known Vocabulary Database")
            print("4. Exit")
            choice = input("Select an option: ")

            if choice == '1':
                filepath = input("Enter path to text file (e.g., chapter1.txt): ")
                self.analyze_text(filepath)
            elif choice == '2':
                filepath = input("Enter path to 50k frequency list: ")
                self.import_50k_frequency(filepath)
            elif choice == '3':
                filepath = input("Enter path to known vocab file: ")
                self.load_known_vocab(filepath)
            elif choice == '4':
                self.conn.close()
                print("Exiting engine. Good luck.")
                sys.exit()
            else:
                print("Invalid choice.")

if __name__ == "__main__":
    engine = LinguisticEngine()
    engine.run_cli()