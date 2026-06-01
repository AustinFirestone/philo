import soundfile as sf
from kokoro import KPipeline

print("Initializing Lean Kokoro Speech Engine...")
# Explicitly use espeak pack for Russian processing
# pipeline = KPipeline(lang_code='r', phonemizer='espeak')
pipeline = KPipeline(lang_code='a', model=False)

text = "Всё, что случилось со мной, было предрешено моей судьбой."

print("Synthesizing studio-grade Russian audio stream...")
generator = pipeline(text, voice='rm_ruslan', speed=1.0)

for i, (gs, ps, audio) in enumerate(generator):
    sf.write("russian_output.wav", audio, 24000)
    print("\n--- Success! ---")
    print("Generated file: 'russian_output.wav' is ready!")