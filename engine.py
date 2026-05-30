import sqlite3
import datetime
from collections import Counter
import sys
import os

try:
    import spacy
    from spacy.cli import download as spacy_download
except ImportError as exc:
    print("Error: spaCy is not installed.")
    print("Run this in your terminal: pip install spacy")
    sys.exit(1)

class LinguisticEngine:
    def __init__(self, db_name="linguist.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self.nlp = self._ensure_russian_model()
        self.setup_database()

    def _ensure_russian_model(self, model_name="ru_core_news_lg"):
        """Attempts to load the model. If missing, automatically downloads it."""
        print(f"Loading NLP Model ({model_name})...")
        try:
            return spacy.load(model_name)
        except OSError:
            print(f"Model not found locally. Automatically downloading '{model_name}'...")
            print("This usually takes a minute but only happens once.")
            try:
                spacy_download(model_name)
                return spacy.load(model_name)
            except Exception as e:
                print(f"Critical error downloading model: {e}")
                sys.exit(1)

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
                interval INTEGER,
                repetitions INTEGER,
                ease_factor REAL,
                next_review DATE
            );
        ''')
        self.conn.commit()

    def import_50k_frequency(self, filepath="ru_50k.txt"):
        """Ingests the 50,000 word frequency list into the database."""
        if not os.path.exists(filepath):
            print(f"\nError: {filepath} not found.")
            print("Run this command in the terminal to download it:")
            print("wget https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/ru/ru_50k.txt")
            return
        
        print("\nImporting 50,000 word frequency list... this takes a few seconds.")
        self.cursor.execute('DELETE FROM global_frequency') 
        
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for rank, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) >= 1:
                    lemma = parts[0].lower()
                    data.append((lemma, rank))
        
        self.cursor.executemany('''
            INSERT OR IGNORE INTO global_frequency (lemma, rank) VALUES (?, ?)
        ''', data)
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
        
        self.cursor.executemany('''
            INSERT OR IGNORE INTO known_vocabulary (lemma, added_date) VALUES (?, ?)
        ''', data)
        self.conn.commit()
        print(f"Successfully imported {len(words)} words into Known Vocabulary.")

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
        
        # Matrix Thresholds
        HIGH_BOOK_FREQ = 2 
        HIGH_RU_FREQ_RANK = 10000 

        for lemma, book_count in unknown_lemmas.items():
            self.cursor.execute('SELECT rank FROM global_frequency WHERE lemma = ?', (lemma,))
            result = self.cursor.fetchone()
            global_rank = result[0] if result else 999999
            
            is_high_book = book_count >= HIGH_BOOK_FREQ
            is_high_ru = global_rank <= HIGH_RU_FREQ_RANK

            if is_high_book and is_high_ru: q1.append(lemma)
            elif not is_high_book and is_high_ru: q2.append(lemma)
            elif is_high_book and not is_high_ru: q3.append(lemma)
            else: q4.append(lemma)

        print(f"\n=== TEXT ANALYTICS REPORT ===")
        print(f"Total words (tokens): {len(lemmas)}")
        print(f"Unique words (lemmas): {len(lemma_counts)}")
        print(f"Known words: {len(lemma_counts) - len(unknown_lemmas)}")
        print(f"Unknown words: {len(unknown_lemmas)}")

        print("\n--- 4-QUADRANT DISTRIBUTION ---")
        print(f"Q1 (High Book / High RU): {len(q1)} words -> {q1[:5]}...")
        print(f"Q2 (Low Book  / High RU): {len(q2)} words -> {q2[:5]}...")
        print(f"Q3 (High Book / Low RU):  {len(q3)} words -> {q3[:5]}...")
        print(f"Q4 (Low Book  / Low RU):  {len(q4)} words -> {q4[:5]}...")
        print("-------------------------------")

        if q1 or q2:
            choice = input(f"\nGenerate flashcards for the {len(q1) + len(q2)} words in Q1 & Q2? (y/n): ")
            if choice.lower() == 'y':
                self.generate_flashcards(q1 + q2)

    def generate_flashcards(self, words):
        today_str = datetime.date.today().isoformat()
        data = [(word, 0, 0, 2.5, today_str) for word in words]
        
        self.cursor.executemany('''
            INSERT OR IGNORE INTO flashcards (lemma, interval, repetitions, ease_factor, next_review) 
            VALUES (?, ?, ?, ?, ?)
        ''', data)
        self.conn.commit()
        print(f"Added {len(words)} words to your active flashcard deck.")

    def review_flashcards(self):
        today_str = datetime.date.today().isoformat()
        
        self.cursor.execute('''
            SELECT lemma, interval, repetitions, ease_factor FROM flashcards 
            WHERE next_review <= ?
        ''', (today_str,))
        due_cards = self.cursor.fetchall()
        
        if not due_cards:
            print("\nYou are completely caught up! No flashcards due today.")
            return
            
        print(f"\n--- You have {len(due_cards)} cards due for review ---")
        
        for card in due_cards:
            lemma, interval, repetitions, ease_factor = card
            print(f"\nWord: {lemma}")
            input("Press Enter to reveal/grade...")
            grade_input = input("Grade (0-5, 5=Perfect, 0=Blackout, 'q' to quit): ")
            
            if grade_input.lower() == 'q':
                break
            
            try:
                grade = int(grade_input)
                if grade < 0 or grade > 5: continue
            except ValueError:
                continue
            
            if grade >= 3:
                if repetitions == 0:
                    interval = 1
                elif repetitions == 1:
                    interval = 6
                else:
                    interval = int(round(interval * ease_factor))
                repetitions += 1
            else:
                repetitions = 0
                interval = 1
            
            ease_factor = ease_factor + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
            if ease_factor < 1.3:
                ease_factor = 1.3
                
            next_review_date = (datetime.date.today() + datetime.timedelta(days=interval)).isoformat()
            
            self.cursor.execute('''
                UPDATE flashcards 
                SET interval=?, repetitions=?, ease_factor=?, next_review=?
                WHERE lemma=?
            ''', (interval, repetitions, ease_factor, next_review_date, lemma))
            
            print(f"Scheduled for {interval} day(s) from now.")
            
        self.conn.commit()
        print("\nReview session saved to database.")

    def run_cli(self):
        while True:
            print("\n[ SOVEREIGN LINGUIST - v0.2 ]")
            print("1. Seed Database (50k Frequency List)")
            print("2. Import Known Vocabulary List")
            print("3. Analyze Text File & Generate Deck")
            print("4. Review Flashcards (SM2)")
            print("5. Exit")
            choice = input("Select an option: ")

            if choice == '1':
                self.import_50k_frequency()
            elif choice == '2':
                filepath = input("Enter path to known vocab file: ")
                self.load_known_vocab(filepath)
            elif choice == '3':
                filepath = input("Enter path to text file: ")
                self.analyze_text(filepath)
            elif choice == '4':
                self.review_flashcards()
            elif choice == '5':
                self.conn.close()
                sys.exit()
            else:
                print("Invalid choice.")

if __name__ == "__main__":
    engine = LinguisticEngine()
    engine.run_cli()