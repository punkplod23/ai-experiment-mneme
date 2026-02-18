import os
import requests
import re
import time

BOOKS_DIR = "./books"

# Expanded 65-book Library with Level 5 (Modern/Specialized)
BOOKS = {
    "dictionary.txt": "29765",
    # Label 1: Rhythmic/Ancient/Structural
    "kjv.txt": "10", "shakespeare.txt": "100", "aurelius_meditations.txt": "1586",
    "plato_republic.txt": "197", "aristotle_ethics.txt": "730", "dante_inferno.txt": "8800",
    "homer_iliad.txt": "6130", "homer_odyssey.txt": "1633", "milton_paradise.txt": "26",
    "epictetus_enchiridion.txt": "4514", "hesiod_theogony.txt": "348",
    
    # Label 2: Philosophy/Logic/Momentum
    "leibniz_theodicy.txt": "17147", "leibniz_essays.txt": "40957", "spinoza_ethics.txt": "3800",
    "kant_reason.txt": "4280", "russell_mind.txt": "2529", "descartes_method.txt": "161",
    "schopenhauer_essays.txt": "3600", "james_psychology.txt": "38103", "nietzsche_zarathustra.txt": "20203",
    "boole_logic.txt": "51711", "mill_liberty.txt": "815", "mill_logic.txt": "27942",
    "machiavelli_prince.txt": "2891", "pearson_grammar.txt": "31872", "locke_understanding.txt": "10615",
    "hume_human.txt": "4320",

    # Label 3: Science/Technical/Hard Physics
    "faraday_candle.txt": "14474", "faraday_electricity.txt": "14986", "newton_opticks.txt": "33504",
    "newton_principia.txt": "28233", "darwin_origin.txt": "1228", "priestley_oxygen.txt": "16053",
    "huygens_light.txt": "43503", "davinci_notebooks.txt": "2500", "steam_engine.txt": "10662", 
    "telegraph.txt": "16812", "industrial_history.txt": "19033", "tyndall_faraday.txt": "1225", 
    "hooke_micrographia.txt": "15491", "boyle_chymist.txt": "22914",

    # Label 4: Vocabulary Breadth (Prose Bridge)
    "melville_moby.txt": "2701", "austen_pride.txt": "1342", "shelley_frankenstein.txt": "84",
    "dickens_cities.txt": "98", "doyle_holmes.txt": "1661", "carroll_alice.txt": "11",
    "thoreau_walden.txt": "205", "joyce_ulysses.txt": "4300", "kafka_metamorphosis.txt": "5200",
    "london_wild.txt": "215", "dickens_expectations.txt": "1400", "stevenson_island.txt": "120",
    "wells_time_machine.txt": "35", "stoker_dracula.txt": "345", "verne_moon.txt": "83",

    # Label 5: Modern/Specialized/Theoretical
    "einstein_relativity.txt": "5001", "curie_radioactivity.txt": "10561", "lorentz_electrons.txt": "39501",
    "heaviside_electromagnetic.txt": "53000", "planck_heat_radiation.txt": "40030",
    "poincare_science_hypothesis.txt": "37157", "boltzmann_gas_theory.txt": "49500",
    "rutherford_radioactive.txt": "45000", # Added missing comma

    # Label 6: Dialogues, Interviews, and Interrogative Logic
    "galileo_dialogue.txt": "43503", 
    "berkeley_dialogues.txt": "4723", 
    "plato_protagoras.txt": "1591",
    "hume_dialogues.txt": "4583", 
    "dryden_dramatic_poesy.txt": "19600", # Added missing comma

    # Label 7: Computation and Coding Logic
    "boole_laws_of_thought.txt": "51711",
    "pearson_grammar_science.txt": "31872",
    "babbage_calculating_engines.txt": "41523",
    "lovelace_notes.txt": "58139",
}

def clean_gutenberg_text(text):
    start_pattern = re.compile(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .* \*\*\*", re.IGNORECASE)
    end_pattern = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .* \*\*\*", re.IGNORECASE)
    start_match = start_pattern.search(text)
    end_match = end_pattern.search(text)
    if start_match and end_match:
        return text[start_match.end():end_match.start()].strip()
    return text

def download_library():
    if not os.path.exists(BOOKS_DIR):
        os.makedirs(BOOKS_DIR)
    for filename, book_id in BOOKS.items():
        file_path = os.path.join(BOOKS_DIR, filename)
        if os.path.exists(file_path): continue
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        print(f"Downloading {filename}...")
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(clean_gutenberg_text(r.text))
            time.sleep(1.5)
        except Exception as e:
            print(f"Error {filename}: {e}")

if __name__ == "__main__":
    download_library()
    print("\nLevel 5 Library Complete.")