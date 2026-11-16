"""bench_llm.py - check LLMs on realistic dictations (correctness + latency).

    python tests/bench_llm.py                    # the model picked in the app's LLM menu (default gemma4:12b-mlx)
    python tests/bench_llm.py gemma4:12b-mlx gemma4:26b    # models on the server picked in the menu

Uses the app's prompt and its own small glossary (GLOSSARY below). Each model is unloaded afterwards.
Exits with status 1 if any case fails, so it doubles as a regression test after
editing SYSTEM_PROMPT.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import tempfile  # noqa: E402

from cleanup import DEFAULT_LLM, Cleaner, Glossary  # noqa: E402
from menubar import load_settings  # noqa: E402

SETTINGS = load_settings()
GLOSSARY = """Ollama: o lama, olama, o llama
MLX: emme elle ics, em el ex
Parakeet
Gemma
Claude Code: cloud code
Claude
Python
GitHub: git hub
"""

# (name, dictated text, check(output) -> bool[, the menu's Style, default "clean"])
CASES = [
    ("correction-it", "Ci vediamo martedì alle cinque, no cancella, mercoledì alle cinque.",
     lambda o: "mercoledì" in o and "martedì" not in o and "cancella" not in o),
    ("anzi", "Prenota il tavolo per otto persone, anzi per dieci, alle nove di sera.",
     lambda o: "dieci" in o and "otto" not in o and "anzi" not in o),
    ("correction-en", "Send the report to Luca by Friday, actually no, by Thursday.",
     lambda o: "Thursday" in o and "Friday" not in o),
    ("translate", "Domani devo finire la relazione per il corso di machine learning e mandarla al professore. "
     "Scrivilo in inglese.",
     lambda o: "report" in o and "professor" in o and "inglese" not in o.lower()),
    ("traduci", "Ciao Marco, ti scrivo per dirti che la riunione di giovedì è spostata a venerdì mattina alle dieci, "
     "fammi sapere se per te va bene. Traduci in inglese.",
     lambda o: "Friday" in o and "meeting" in o and "Traduci" not in o),
    ("glossary", "Ho installato o lama e uso il modello emme elle ics per la dettatura.",
     lambda o: "Ollama" in o and "MLX" in o),
    ("spoken-punct", "Caro Marco virgola a capo ti scrivo per confermare la riunione di domani punto",
     lambda o: o.startswith("Caro Marco,\n") and "a capo" not in o and o.endswith(".")),
    ("a-capo-after-word", "Prompt a capo devo creare un'app in React e in Python che mi permetta di collegarmi "
     "alle API di Claude o Gemini utilizzando i nuovi modelli.",
     lambda o: o.startswith("Prompt\n") and "a capo" not in o),
    ("keep-label", "Prompt a capo scrivi una mail al professore.",
     lambda o: o.startswith("Prompt\n") and "a capo" not in o),
    ("glossary-context", "Devo creare un'app in React che si collega all'API di Cloud o Gemini. "
     "Traducimelo in inglese.",
     lambda o: "Claude" in o and "Gemini" in o and "Traduc" not in o),
    ("translate-addressed", "Domani vado in stazione a prendere il treno per Firenze e poi torno la sera tardi. "
     "Ok Gemma, perfetto, ora scrivimi tutto quanto in inglese.",
     lambda o: "train" in o and "Florence" in o and "Gemma" not in o and "stazione" not in o),
    ("translate-next-only", "Devo andare a Verona, anzi no, devo andare a Milano. Solo la prossima frase traducimela "
     "in inglese. Dopo Milano prendo un treno per tornare a Pisa.",
     lambda o: o.startswith("Devo andare a Milano") and "Verona" not in o and "train" in o and "traduc" not in o),
    ("unchanged","Allora, secondo me la soluzione migliore è usare un modello piccolo, perché deve essere veloce.",
     lambda o: o == "Allora, secondo me la soluzione migliore è usare un modello piccolo, perché deve essere veloce."),
    ("dont-answer", "Puoi spiegarmi come funziona il meccanismo di attenzione nei transformer?",
     lambda o: o.startswith("Puoi spiegarmi") and len(o) < 120),
    ("prompt-dictation", "Scrivi una funzione Python che legge un file CSV e restituisce la media della colonna "
     "prezzo, gestisci anche il caso in cui il file non esiste.",
     lambda o: o.startswith("Scrivi una funzione Python") and "def " not in o),
    ("long-mid-correction", "Allora, per il progetto di dettatura vorrei aggiungere un secondo passaggio con un "
     "modello linguistico locale che sistemi il testo dopo la trascrizione. Deve essere veloce, quindi al massimo "
     "uno o due secondi, e deve funzionare tutto in locale senza cloud. Poi vorrei anche un glossario con i nomi "
     "dei professori e dei corsi, no cancella, con i nomi dei corsi e degli strumenti che uso, tipo o lama e emme "
     "elle ics.",
     lambda o: "professori" not in o and "cancella" not in o and "Ollama" in o and "MLX" in o),
    ("style-formal", "Ciao Marco, guarda, domani non ce la faccio proprio a venire alla riunione, ci sentiamo dopo.",
     lambda o: "Marco" in o and "riunione" in o and "guarda" not in o and "non ce la faccio proprio" not in o
     and len(o) < 250, "formal"),
    ("style-bullets", "Per la cena servono tre cose. Il pane, il vino rosso e un dolce al cioccolato.",
     lambda o: sum(line.startswith("- ") for line in o.splitlines()) >= 3 and "vino rosso" in o, "bullets"),
    ("style-english", "Domani mattina vado in ufficio presto perché ho una riunione con il cliente.",
     lambda o: "office" in o and "meeting" in o and "riunione" not in o, "english"),
    ("style-english-correction", "Ci vediamo martedì alle cinque, no cancella, mercoledì alle cinque.",
     lambda o: "Wednesday" in o and "Tuesday" not in o and "cancel" not in o.lower(), "english"),
    ("style-english-already", "Send the report to Luca by Friday, actually no, by Thursday.",
     lambda o: "Thursday" in o and "Friday" not in o and "report" in o, "english"),
]


def bench(model: str, glossary: Glossary) -> int:
    c = Cleaner(model, "", glossary, timeout=30, provider=SETTINGS["llm_provider"])
    warning = c.check()
    if warning:
        print(f"⚠️  {warning}")
        return len(CASES)
    t0 = time.perf_counter()
    c.warm_up()
    print(f"\n#### {model}  (load + warm-up {time.perf_counter() - t0:.1f}s)")
    failures = 0
    for name, text, ok, *style in CASES:
        t0 = time.perf_counter()
        out = c.clean(glossary.apply(text), style=(style or ["clean"])[0])
        dt = time.perf_counter() - t0
        good = out is not None and ok(out)
        failures += not good
        print(f"{'✓' if good else '✗'} {name:20s} {dt:5.2f}s  {out!r}" + ("" if out is not None else c.last_error))
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    c.unload()
    return failures


if __name__ == "__main__":
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(GLOSSARY)
    failures = sum(bench(m, Glossary(f.name)) for m in (sys.argv[1:] or [SETTINGS["llm_model"] or DEFAULT_LLM]))
    os.remove(f.name)
    sys.exit(1 if failures else 0)
