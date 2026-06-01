import os

def extract_russian_column(input_file, output_file="extracted_for_known.txt"):
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return

    russian_words = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Anki files are tab-separated; split by '\t'
            parts = line.strip().split('\t')
            if parts and parts[0]:
                russian_words.append(parts[0].strip())

    if not russian_words:
        print("No words found. Is the file formatted correctly?")
        return

    # Write the clean column out to a text file
    with open(output_file, 'w', encoding='utf-8') as out_f:
        out_f.write('\n'.join(russian_words) + '\n')

    print(f"\n[+] Extracted {len(russian_words)} words into: {output_file}")
    print("You can now safely copy-paste these contents directly into your known_words.txt!")

if __name__ == "__main__":
    # Change 'chapter1_anki_deck.txt' to whatever your file is named
    filename = input("Enter your Anki deck filename (e.g., chapter1_anki_deck.txt): ")
    extract_russian_column(filename)