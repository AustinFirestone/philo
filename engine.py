import sqlite3
import datetime
from collections import Counter
import sys
import os

try:
    import spacy
except ImportError as exc:
    print("Error: spaCy is not installed or cannot be imported.")
    print("Install dependencies with: python3 -m pip install -r requirements.txt")
    raise SystemExit(1) from exc

class LinguisticEngine:
    def __init__(self, db_name="linguist.db"):
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        print("Loading Russian NLP Model (ru_core_news_lg)... This takes a moment.")
        self.nlp = self._load_russian_model()
        self.setup_database()
        self.seed_mock_frequency() # Temporary: adds mock data until we get the 50k list

    def _load_russian_model(self):
        try:
            return spacy.load("ru_core_news_lg")
        except OSError:
            try:
                return spacy.load("ru_core_news_sm")
            except OSError:
                print("Warning: spaCy Russian model not found.")
                print("Using a lightweight blank Russian pipeline instead.")
                print("For better results, install the model with:")
                print("  python3 -m pip install ru_core_news_lg")
                print("or run: python3 -m spacy download ru_core_news_lg")
                return spacy.blank("ru")

    def setup_database(self):
        """Initializes the SQLite schema if it doesn't exist."""
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

    def seed_mock_frequency(self):
        """Seeds a tiny frequency list so the quadrant math works on day one."""
        mock_data = [("и", 1), ("в", 2), ("не", 3), ("он", 4), ("я", 5), ("быть", 6), 
                     ("человек", 50), ("один", 55), ("парус", 2500), ("одинокий", 4000), 
                     ("туман", 5500), ("море", 600)]
        self.cursor.executemany('''
            INSERT OR IGNORE INTO global_frequency (lemma, rank) VALUES (?, ?)
        ''', mock_data)
        self.conn.commit()

    def load_known_vocab(self, filepath):
        """Bulk imports known words from a text file."""
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
        """Parses text, lemmatizes, and sorts into the 4 Quadrants."""
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found.")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        doc = self.nlp(text)
        # Extract valid alphabetical lemmas, ignoring pure punctuation
        lemmas = [token.lemma_.lower() for token in doc if token.is_alpha]
        lemma_counts = Counter(lemmas)
        
        # Cross-reference with known vocab
        self.cursor.execute('SELECT lemma FROM known_vocabulary')
        known_set = set(row[0] for row in self.cursor.fetchall())
        
        unknown_lemmas = {lemma: count for lemma, count in lemma_counts.items() if lemma not in known_set}
        
        print(f"\n=== TEXT ANALYTICS REPORT ===")
        print(f"Total words (tokens): {len(lemmas)}")
        print(f"Unique words (lemmas): {len(lemma_counts)}")
        print(f"Known words: {len(lemma_counts) - len(unknown_lemmas)}")
        print(f"Unknown words: {len(unknown_lemmas)}\n")

        # Sort into Quadrants
        q1, q2, q3, q4 = [], [], [], []
        
        # Arbitrary thresholds for Phase 1 testing
        HIGH_BOOK_FREQ = 2 
        HIGH_RU_FREQ_RANK = 5000 

        for lemma, book_count in unknown_lemmas.items():
            self.cursor.execute('SELECT rank FROM global_frequency WHERE lemma = ?', (lemma,))
            result = self.cursor.fetchone()
            global_rank = result[0] if result else 999999 # Treat missing as extremely rare
            
            is_high_book = book_count >= HIGH_BOOK_FREQ
            is_high_ru = global_rank <= HIGH_RU_FREQ_RANK

            if is_high_book and is_high_ru: q1.append(lemma)
            elif not is_high_book and is_high_ru: q2.append(lemma)
            elif is_high_book and not is_high_ru: q3.append(lemma)
            else: q4.append(lemma)

        print("--- 4-QUADRANT DISTRIBUTION ---")
        print(f"Q1 (High Book / High RU): {len(q1)} words -> {q1[:5]}...")
        print(f"Q2 (Low Book  / High RU): {len(q2)} words -> {q2[:5]}...")
        print(f"Q3 (High Book / Low RU):  {len(q3)} words -> {q3[:5]}...")
        print(f"Q4 (Low Book  / Low RU):  {len(q4)} words -> {q4[:5]}...")
        print("-------------------------------\n")

    def run_cli(self):
        """The main interactive dashboard."""
        while True:
            print("\n[ SOVEREIGN LINGUIST - v0.1 ]")
            print("1. Analyze Text File")
            print("2. Import Known Vocabulary List")
            print("3. Exit")
            choice = input("Select an option: ")

            if choice == '1':
                filepath = input("Enter path to text file (e.g., lermontov.txt): ")
                self.analyze_text(filepath)
            elif choice == '2':
                filepath = input("Enter path to known vocab file (e.g., known.txt): ")
                self.load_known_vocab(filepath)
            elif choice == '3':
                print("Exiting engine. Shutting down database connection.")
                self.conn.close()
                sys.exit()
            else:
                print("Invalid choice.")

if __name__ == "__main__":
    engine = LinguisticEngine()
    engine.run_cli()
