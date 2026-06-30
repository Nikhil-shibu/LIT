from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from truthlens.logging_config import get_logger

log = get_logger("baseline")


def train_xgb_pipeline():
    """Train and return a TF-IDF + XGBoost baseline pipeline on 80 labelled examples."""
    if not XGBOOST_AVAILABLE:
        log.warning("xgboost not installed — baseline pipeline unavailable")
        return None

    genuine_texts = [
        "The capital of France is Paris.",
        "Water freezes at zero degrees Celsius.",
        "The sun rises in the east.",
        "A year has 365 days normally.",
        "Gravity pulls objects towards the earth.",
        "Photosynthesis is how plants make food.",
        "The human body has 206 bones.",
        "Earth revolves around the sun.",
        "Mount Everest is the highest mountain.",
        "Oxygen is essential for human life.",
        "Dogs are mammals.",
        "The Pacific is the largest ocean.",
        "An apple a day keeps the doctor away.",
        "Water is made of hydrogen and oxygen.",
        "Light travels faster than sound.",
        "Humans need to drink water to survive.",
        "The moon orbits the earth.",
        "Rome is the capital of Italy.",
        "Ice is the solid form of water.",
        "Birds have feathers.",
        "Sharks live in the ocean.",
        "Bees produce honey.",
        "The Sahara is a large desert.",
        "Trees produce oxygen.",
        "Fish breathe through gills.",
        "A triangle has three sides.",
        "A minute has sixty seconds.",
        "A day has twenty four hours.",
        "The Great Wall of China is very long.",
        "Jupiter is the largest planet.",
        "Neil Armstrong walked on the moon.",
        "Shakespeare wrote Hamlet.",
        "Iron is a metal.",
        "Cats are popular pets.",
        "The boiling point of water is 100 degrees Celsius.",
        "Diamonds are made of carbon.",
        "The human heart pumps blood.",
        "Python is a programming language.",
        "The earth is round.",
        "Fire is hot.",
    ]

    misleading_texts = [
        "The earth is completely flat.",
        "Vaccines cause widespread autism.",
        "The moon landing was filmed on a soundstage.",
        "Climate change is a hoax.",
        "Drinking bleach cures all viruses.",
        "5G towers spread diseases.",
        "Lizards secretly control the government.",
        "Aliens built the pyramids.",
        "Birds aren't real, they are government drones.",
        "Chemtrails are mind control chemicals.",
        "The sun revolves around the earth.",
        "Eating rocks is healthy.",
        "You can charge your phone in the microwave.",
        "The government hides the cure for aging.",
        "Mermaids have been found in the Atlantic.",
        "Unicorns exist in North Korea.",
        "Pigs can actually fly if trained.",
        "Gravity is an illusion.",
        "The earth is hollow inside.",
        "Elvis is still alive and hiding.",
        "Chocolate milk comes from brown cows.",
        "Water is dry.",
        "The sky is permanently neon green.",
        "Humans don't need oxygen.",
        "Trees talk to each other in English.",
        "Dinosaurs are still alive in Africa.",
        "You can walk to the moon.",
        "Fire is completely cold.",
        "Ice sinks in water.",
        "Rocks are soft until you touch them.",
        "Reading in the dark makes you blind.",
        "Swallowing gum stays in your stomach for 7 years.",
        "Lightning never strikes the same place twice.",
        "Goldfish have a 3-second memory.",
        "Bulls hate the color red.",
        "We only use 10 percent of our brains.",
        "Shaving makes hair grow back thicker.",
        "Cracking your knuckles causes arthritis.",
        "Toads give you warts.",
        "Bats are completely blind.",
    ]

    X = genuine_texts + misleading_texts
    y = [0] * len(genuine_texts) + [1] * len(misleading_texts)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("xgb", XGBClassifier(eval_metric="logloss", random_state=42)),
    ])
    pipeline.fit(X, y)
    log.info(f"XGBoost pipeline trained on {len(X)} examples")
    return pipeline


def xgb_pipeline_predict(text: str, pipeline) -> Dict[str, object]:
    """Run the baseline XGBoost pipeline and return its label and probability."""
    if pipeline is None or not text.strip():
        return {"label": "Genuine", "probability": 0.0}

    proba = pipeline.predict_proba([text])[0]
    prob_genuine, prob_misleading = float(proba[0]), float(proba[1])

    # If the text has no words in the TF-IDF vocabulary, XGBoost outputs ~0.53 misleading.
    # We catch these "uncertain" base-rate outputs and force them to neutral/Genuine
    # so we don't accidentally flag real news just because the words are new.
    if 0.45 < prob_misleading < 0.55:
        log.debug(f"XGBoost output near base rate ({prob_misleading:.3f}) -> forcing to Neutral/Genuine")
        return {"label": "Genuine", "probability": 0.5}

    if prob_misleading > prob_genuine:
        return {"label": "Misleading", "probability": prob_misleading}
    return {"label": "Genuine", "probability": prob_genuine}
