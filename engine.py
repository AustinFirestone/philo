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
        # Migrations: Silently add new columns to existing databases
        try:
            self.cursor.execute("ALTER TABLE flashcards ADD COLUMN translation TEXT")
        except sqlite3.OperationalError:
            pass
            
        try:
            self.cursor.execute("ALTER TABLE flashcards ADD COLUMN last_reviewed DATE")
        except sqlite3.OperationalError:
            pass
            
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
            return
        with open(filepath, 'r', encoding='utf-8') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        today_str = datetime.date.today().isoformat()
        data = [(word, today_str) for word in words]
        self.cursor.executemany('INSERT OR IGNORE INTO known_vocabulary (lemma, added_date) VALUES (?, ?)', data)
        self.conn.commit()
        print(f"Successfully imported {len(words)} words.")

    def analyze_text(self, filepath):
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found.")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        doc = self.nlp(text)
        lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
        lemma_counts = Counter(lemmas)
        
        self.cursor.execute('SELECT lemma FROM known_vocabulary')
        known_set = set(row[0] for row in self.cursor.fetchall())
        unknown_lemmas = {lemma: count for lemma, count in lemma_counts.items() if lemma not in known_set}
        
        q1, q2, q3, q4 = [], [], [], []
        HIGH_BOOK_FREQ = 2 
        HIGH_RU_FREQ_RANK = 10000 

        for lemma, book_count in unknown_lemmas.items():
            self.cursor.execute('SELECT rank FROM global_frequency WHERE lemma = ?', (lemma,))
            result = self.cursor.fetchone()
            global_rank = result[0] if result else 999999
            
            if book_count >= HIGH_BOOK_FREQ and global_rank <= HIGH_RU_FREQ_RANK: q1.append(lemma)
            elif book_count < HIGH_BOOK_FREQ and global_rank <= HIGH_RU_FREQ_RANK: q2.append(lemma)
            elif book_count >= HIGH_BOOK_FREQ and global_rank > HIGH_RU_FREQ_RANK: q3.append(lemma)
            else: q4.append(lemma)

        print(f"\n=== TEXT ANALYTICS REPORT ===")
        print(f"Total words (tokens): {len(lemmas)}")
        print(f"Unique words (lemmas): {len(lemma_counts)}")
        print(f"Known words: {len(lemma_counts) - len(unknown_lemmas)}")
        print(f"Unknown words: {len(unknown_lemmas)}")
        print("\n--- 4-QUADRANT DISTRIBUTION ---")
        print(f"Q1 (High Book / High RU): {len(q1)} words")
        print(f"Q2 (Low Book  / High RU): {len(q2)} words")
        print(f"Q3 (High Book / Low RU):  {len(q3)} words")
        print(f"Q4 (Low Book  / Low RU):  {len(q4)} words")
        print("-------------------------------")

        export_choice = input("\nExport detailed Markdown report? (y/n): ")
        if export_choice.lower() == 'y':
            self.export_report(filepath, len(lemmas), len(lemma_counts), len(known_set), unknown_lemmas, q1, q2, q3, q4)

        if q1 or q2:
            choice = input(f"\nGenerate flashcards for the {len(q1) + len(q2)} words in Q1 & Q2? (y/n): ")
            if choice.lower() == 'y':
                self.generate_flashcards(q1 + q2)

    def export_report(self, original_file, tokens, unique, known, unknown_dict, q1, q2, q3, q4):
        report_name = f"analysis_report_{datetime.date.today().isoformat()}.md"
        with open(report_name, 'w', encoding='utf-8') as f:
            f.write(f"# Linguistic Analysis Report: {original_file}\n\n")
            f.write("## Global Metrics\n")
            f.write(f"- **Total Tokens:** {tokens}\n")
            f.write(f"- **Unique Lemmas:** {unique}\n")
            f.write(f"- **Known Words Avoided:** {unique - len(unknown_dict)}\n")
            f.write(f"- **Unknown Words Remaining:** {len(unknown_dict)}\n\n")
            f.write("## Quadrant 1 (Priority: High Book / High Russian)\n")
            f.write(f"{', '.join(q1) if q1 else 'None'}\n\n")
            f.write("## Quadrant 2 (Core Gaps: Low Book / High Russian)\n")
            f.write(f"{', '.join(q2) if q2 else 'None'}\n\n")
            f.write("## Quadrant 3 (Motifs: High Book / Low Russian)\n")
            f.write(f"{', '.join(q3) if q3 else 'None'}\n\n")
            f.write("## Quadrant 4 (Archaic/Rare: Low Book / Low Russian)\n")
            f.write(f"{', '.join(q4) if q4 else 'None'}\n\n")
        print(f"\n[+] Detailed report exported to: {report_name}")

    def generate_flashcards(self, words):
        today_str = datetime.date.today().isoformat()
        data = []
        print(f"Fetching translations for {len(words)} words... (This may take a moment)")
        
        for word in words:
            try:
                translation = self.translator.translate(word)
            except Exception:
                translation = "Translation failed"
            # Note the extra today_str at the end for 'last_reviewed'
            data.append((word, translation, 0, 0, 2.5, today_str, today_str))
        
        self.cursor.executemany('''
            INSERT OR IGNORE INTO flashcards (lemma, translation, interval, repetitions, ease_factor, next_review, last_reviewed) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data)
        
        # Update existing cards if they were generated without translations/dates previously
        for d in data:
            self.cursor.execute('''
                UPDATE flashcards 
                SET translation = ?, last_reviewed = ? 
                WHERE lemma = ? AND (translation IS NULL OR last_reviewed IS NULL)
            ''', (d[1], d[6], d[0]))
            
        self.conn.commit()
        print(f"Added {len(words)} words with translations to your active deck.")

    def review_flashcards(self):
        today_str = datetime.date.today().isoformat()
        self.cursor.execute('''
            SELECT lemma, translation, interval, repetitions, ease_factor FROM flashcards 
            WHERE next_review <= ?
        ''', (today_str,))
        due_cards = self.cursor.fetchall()
        
        if not due_cards:
            print("\nYou are completely caught up! No flashcards due today.")
            return
            
        print(f"\n--- You have {len(due_cards)} cards due for review ---")
        
        for card in due_cards:
            lemma, translation, interval, repetitions, ease_factor = card
            print(f"\n---------------------------------")
            print(f"Russian: {lemma}")
            input("Press Enter to reveal translation...")
            print(f"English: {translation}")
            print(f"---------------------------------")
            
            grade_input = input("Grade (0-5, 5=Perfect, 0=Blackout, 'q' to quit): ")
            if grade_input.lower() == 'q': break
            
            try:
                grade = int(grade_input)
                if grade < 0 or grade > 5: continue
            except ValueError: continue
            
            if grade >= 3:
                if repetitions == 0:
                    interval = 4 if grade == 5 else 1 
                elif repetitions == 1:
                    interval = 6
                else:
                    interval = int(round(interval * ease_factor))
                repetitions += 1
            else:
                repetitions = 0
                interval = 1
            
            ease_factor = max(1.3, ease_factor + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02)))
            next_review_date = (datetime.date.today() + datetime.timedelta(days=interval)).isoformat()
            
            # Update the stats AND the last_reviewed timestamp
            self.cursor.execute('''
                UPDATE flashcards 
                SET interval=?, repetitions=?, ease_factor=?, next_review=?, last_reviewed=?
                WHERE lemma=?
            ''', (interval, repetitions, ease_factor, next_review_date, today_str, lemma))
            
            print(f"Scheduled for {interval} day(s) from now.")
            
        self.conn.commit()
        print("\nReview session saved to database.")

    def browse_database(self):
        self.cursor.execute('SELECT lemma, translation, interval, repetitions, ease_factor, next_review FROM flashcards ORDER BY next_review ASC')
        cards = self.cursor.fetchall()
        
        if not cards:
            print("\nYour flashcard database is empty.")
            return
            
        print("\n=== FLASHCARD DATABASE BROWSER ===")
        print(f"{'LEMMA'.ljust(15)} | {'TRANSLATION'.ljust(20)} | {'INT(DAYS)'.ljust(10)} | {'REPS'.ljust(5)} | {'EASE'.ljust(5)} | {'NEXT REVIEW'}")
        print("-" * 80)
        for c in cards:
            lemma, trans, interval, reps, ease, next_rev = c
            trans_str = (trans[:17] + "...") if trans and len(trans) > 17 else str(trans)
            print(f"{lemma.ljust(15)} | {trans_str.ljust(20)} | {str(interval).ljust(10)} | {str(reps).ljust(5)} | {f'{ease:.2f}'.ljust(5)} | {next_rev}")
        print("-" * 80)
        print(f"Total Active Cards: {len(cards)}\n")

    def export_daily_audio(self):
        """Dumps all words interacted with today into a clean text file for TTS processing."""
        today_str = datetime.date.today().isoformat()
        
        self.cursor.execute('''
            SELECT lemma, translation FROM flashcards 
            WHERE last_reviewed = ?
        ''', (today_str,))
        daily_words = self.cursor.fetchall()
        
        if not daily_words:
            print("\nYou haven't reviewed or added any words today. Nothing to export.")
            return
            
        filename = f"audio_export_{today_str}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            for lemma, translation in daily_words:
                f.write(f"{lemma} | {translation}\n")
                
        print(f"\n[+] Successfully exported {len(daily_words)} words to {filename}")
        print("Format: russian_word | english_translation (ready for Python TTS script parsing).")

    def run_cli(self):
        while True:
            print("\n[ SOVEREIGN LINGUIST - v0.4 ]")
            print("1. Analyze Text File & Generate Deck")
            print("2. Review Flashcards (SM2)")
            print("3. Browse Flashcard Database")
            print("4. Import Known Vocabulary List")
            print("5. Export Daily Audio File (TTS)")
            print("6. Exit")
            choice = input("Select an option: ")

            if choice == '1':
                filepath = input("Enter path to text file: ")
                self.analyze_text(filepath)
            elif choice == '2':
                self.review_flashcards()
            elif choice == '3':
                self.browse_database()
            elif choice == '4':
                filepath = input("Enter path to known vocab file: ")
                self.load_known_vocab(filepath)
            elif choice == '5':
                self.export_daily_audio()
            elif choice == '6':
                self.conn.close()
                sys.exit()
            else:
                print("Invalid choice.")

if __name__ == "__main__":
    engine = LinguisticEngine()
    engine.run_cli()