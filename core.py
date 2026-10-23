"""FairForge core logic.

What happens to one resume:
  1. HIDE IDENTITY      spaCy + rules remove names, places, colleges, pronouns, contact details and graduation years.
  2. SORT SECTIONS      a CNN labels every line: Summary / Skills / Experience / Education / Other.
  3. READ THE CAREER    a BiLSTM (a type of RNN) reads the job-history lines in order and scores the career path.
  4. JUDGE THE FIT      MiniLM turns the resume into meaning numbers and a DNN gives a job-fit probability.
  5. SCORE AND CHECK    a transparent score mixes the signals, SHAP explains it, the Twin Test checks it for fairness.

main.py only talks to the FairForge class at the bottom of this file.
"""
import collections
import json
import re
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

# Folder with the trained model files; it sits next to this file.
MODEL_DIR = Path(__file__).resolve().parent / "models"
# The six labels the section CNN can give to a line (same order as in training).
SECTION_LABELS = ["HEADER", "SUMMARY", "SKILLS", "EXPERIENCE", "EDUCATION", "OTHER"]
MINILM_VECTOR_SIZE = 384          # size of one MiniLM meaning vector
CAREER_VECTOR_SIZE = 128          # size of the BiLSTM career vector (2 directions x 64)

# ------------------------------------------------------------------------------------------- text helpers
# Splits text into words. Identity tags such as [name] stay whole, and characters used in skill names (c++, c#, node.js)
# are kept.
WORD_PATTERN = re.compile(r"\[[a-z]+\]|[a-z0-9+#.]+")      # keeps tags such as [name] as one word


def squash_whitespace(text):
    """Turn every run of spaces, tabs and new lines into a single space."""
    return re.sub(r"\s+", " ", str(text)).strip()


def tokenize(text):
    """Lower-case words of a text (identity tags like [name] stay in one piece)."""
    return WORD_PATTERN.findall(str(text).lower())


# The neural networks cannot read words, so every word is swapped for its number from vocab.json (0 = padding, 1 =
# unknown).
def encode_to_word_ids(texts, vocabulary, max_length):
    """Turn lines of text into a padded matrix of word numbers (0 = padding, 1 = unknown word)."""
    word_id_matrix = np.zeros((len(texts), max_length), dtype=np.int64)
    for row_index, text in enumerate(texts):
        word_ids = [vocabulary.get(word, 1) for word in tokenize(text)][:max_length]
        word_id_matrix[row_index, :len(word_ids)] = word_ids
    return word_id_matrix


# Turns raw network outputs into probabilities that add up to 1.
def softmax(scores):
    """Turn any list of scores into probabilities that add up to 1."""
    scores = np.asarray(scores, dtype=np.float64)
    exponentials = np.exp(scores - scores.max())
    return exponentials / exponentials.sum()


# A resume line is the unit the section CNN works on. The 400 line cap protects against huge files.
def split_into_lines(text):
    """Split a resume into lines. If the text has almost no line breaks, split it into sentences instead."""
    lines = [squash_whitespace(line) for line in str(text).splitlines()]
    lines = [line for line in lines if len(line) > 1]
    if len(lines) < 8:
        lines = [squash_whitespace(part) for part in re.split(r"(?<=[.;])\s+", squash_whitespace(text)) if len(part) > 1]
    return lines[:400]


# ------------------------------------------------------------------------------------------- the three neural networks
# Attribute names (emb, convs, fc, lstm, net ...) must stay as they are: they match the saved model files.
# Step 2 of the pipeline. Input: one line of the resume. Output: the section it belongs to (see SECTION_LABELS).
class SectionCNN(nn.Module):
    """CNN: slides small windows (2, 3 and 4 words wide) over a line to spot which section it belongs to."""

    def __init__(self, vocab_size, num_classes, embedding_size=100, num_filters=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_size, num_filters, width, padding=width // 2) for width in (2, 3, 4)])
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(num_filters * 3, num_classes)

    def forward(self, word_ids):
        word_vectors = self.emb(word_ids).transpose(1, 2)
        pooled_features = torch.cat([torch.relu(conv(word_vectors)).max(dim=2).values for conv in self.convs], dim=1)
        return self.fc(self.drop(pooled_features))


# Step 3. Input: the job-history lines, in order. Output: how well the career path fits each job family.
class CareerBiLSTM(nn.Module):
    """RNN (BiLSTM): reads the job-history lines in order, forwards and backwards."""

    def __init__(self, vocab_size, num_classes, embedding_size=64, hidden_size=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def encode(self, word_ids):
        """word_ids has shape [batch, lines, words]. Returns one career vector per resume."""
        word_mask = (word_ids > 0).float().unsqueeze(-1)
        line_vectors = (self.emb(word_ids) * word_mask).sum(2) / word_mask.sum(2).clamp(min=1)    # average of the words in each line
        lstm_output, _ = self.lstm(line_vectors)
        line_mask = (word_ids.sum(2) > 0).float().unsqueeze(-1)                                   # ignore empty (padding) lines
        return (lstm_output * line_mask).sum(1) / line_mask.sum(1).clamp(min=1)

    def forward(self, word_ids):
        return self.fc(self.drop(self.encode(word_ids)))


# Step 4. Input: the MiniLM meaning vector (plus optional section shares and career vector). Output: a fit probability
# per job family.
class FitDNN(nn.Module):
    """DNN: three hidden layers that turn the resume features into a fit probability for each job family."""

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, 512), nn.ReLU(), nn.Dropout(0.4),
                                 nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, features):
        return self.net(features)


# ------------------------------------------------------------------------------------------- identity hiding
# Patterns for details that have a fixed shape. RULES below pairs each pattern with the tag that replaces it.
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
PRONOUN = re.compile(r"\b(?:he|she|him|her|his|hers|himself|herself)\b", re.I)
TITLE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss)\.?(?=\s)")
# College names can reveal background, so they are hidden as well.
COLLEGE = re.compile(r"\b(?:[A-Z][\w&'.-]*\s+){0,4}(?:University|College|Institute)(?:\s+of\s+(?:[A-Z][\w&'.-]*\s*){1,4})?")
NAME_SHAPE = re.compile(r"[A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+)*")       # "Priya", "Priya Nair", "Priya K. Nair"
# Regex rules applied to every line, in this order.
RULES = [(EMAIL, "[EMAIL]"), (URL, "[URL]"), (PHONE, "[PHONE]"), (COLLEGE, "[COLLEGE]"), (PRONOUN, "[PRONOUN]"), (TITLE, "[TITLE]")]

# Common first names. A name is hidden when spaCy says "person" AND the first word is in this list.
FIRST_NAMES = set("""emily sarah jessica ashley amanda jennifer lisa mary linda patricia susan karen nancy laura emma olivia sophia hannah megan
rachel lauren anna maria michelle kimberly donna carol sandra ruth sharon deborah cynthia angela melissa brenda amy katherine christine
samantha nicole elizabeth stephanie rebecca heather james john robert michael william david richard joseph thomas charles christopher
daniel matthew anthony mark donald steven paul andrew joshua kenneth kevin brian george edward ronald timothy jason jeffrey ryan jacob
gary nicholas eric jonathan stephen larry justin scott brandon benjamin samuel gregory alexander patrick frank raymond jack dennis jerry
tyler aaron jose adam nathan henry peter rahul priya amit anil sunil suresh ramesh rajesh vijay sanjay ajay manoj deepak arun ravi kiran
anita sunita sita geeta meena lakshmi divya pooja neha kavita anjali swati rohit vikram arjun karthik krishna ganesh mohan naveen prakash
srinivas venkat aditya naman jeyaditya prasath ayesha fatima mohammed muhammad ahmed ali omar imran zainab nikhil varun siddharth harish
mahesh pradeep sandeep ashok rakesh vinod anand shruti nisha sneha preeti rekha usha radha shweta""".split())
# Words that spaCy sometimes mistakes for names or places (skills, tools, job words). They are never hidden.
NOT_A_NAME = set("""active directory microsoft office visual studio customer service services project management business data systems system
network web design sales account accounting financial health care medical human resources quality control assurance software hardware
technical support help desk team lead senior junior general operations administrative executive public relations social media oracle java
python cisco windows linux client""".split())
NOT_A_PLACE = set("""cisco windows linux client mac oracle java visio twitter facebook linkedin oversaw managed maintained developed supervised led
trained assisted coordinated responsible performed handled provided created implemented""".split())


# Step 1 of the pipeline. Names and places come from spaCy, everything else from the regex rules above.
class Anonymizer:
    """Hides names, places, colleges, pronouns and contact details (and graduation years once the sections are known)."""

    def __init__(self):
        self.nlp = None
        try:
            # spaCy is optional. Without its English model the regex rules still work, only name detection gets weaker.
            import spacy
            self.nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
        except Exception:
            self.nlp = None          # the rules still work; the app shows a warning that name detection is reduced

    @property
    def ner_available(self):
        """True when spaCy's English model could be loaded."""
        return self.nlp is not None

    # Main method of the anonymizer. For every line it asks spaCy for names and places, replaces them, then applies the
    # regex rules.
    def hide_lines(self, lines):
        """Return (hidden lines, counter of what was hidden). Names in the first 3 lines are hidden even if the first name is unknown."""
        hidden_lines, hidden_counts = [], collections.Counter()
        spacy_docs = list(self.nlp.pipe(lines, batch_size=64)) if self.nlp is not None else [None] * len(lines)
        for line_index, (line, spacy_doc) in enumerate(zip(lines, spacy_docs)):
            working_text = line
            if spacy_doc is not None:
                replacements = []                                         # (start, end, placeholder) to apply
                for entity in spacy_doc.ents:
                    entity_words = entity.text.replace(".", " ").split()
                    placeholder = None
                    if entity.label_ == "PERSON" and entity_words:
                        # A PERSON entity is hidden only when it really looks like a name: the first word is a known
                        # first name, or it stands at the top of the resume. This keeps skills such as Active Directory
                        # from being hidden by mistake.
                        is_known_first_name = (entity_words[0].lower() in FIRST_NAMES
                                               and not any(word.lower() in NOT_A_NAME for word in entity_words[1:]))
                        is_name_at_top = (line_index < 3 and entity.start_char < 60 and NAME_SHAPE.fullmatch(entity.text.strip()) is not None
                                          and not any(word.lower() in NOT_A_NAME for word in entity_words))
                        placeholder = "[NAME]" if (is_known_first_name or is_name_at_top) else None
                    # GPE means country, city or state. Tool names that spaCy often mistakes for places are skipped.
                    elif entity.label_ == "GPE" and entity.text.strip().lower() not in NOT_A_PLACE:
                        placeholder = "[PLACE]"
                    if placeholder:
                        replacements.append((entity.start_char, entity.end_char, placeholder))
                for start_char, end_char, placeholder in sorted(replacements, reverse=True):      # right to left keeps positions valid
                    working_text = working_text[:start_char] + placeholder + working_text[end_char:]
                    hidden_counts[placeholder] += 1
            for pattern, placeholder in RULES:
                working_text, how_many = pattern.subn(placeholder, working_text)
                hidden_counts[placeholder] += how_many
            hidden_lines.append(working_text)
        return hidden_lines, hidden_counts

    # A graduation year can reveal age, so years are hidden only on Education lines. Job dates stay visible.
    @staticmethod
    def hide_years(line, hidden_counts):
        """Hide four-digit years (used only on Education lines, so job dates stay readable)."""
        line, how_many = YEAR.subn("[YEAR]", line)
        hidden_counts["[YEAR]"] += how_many
        return line


# ------------------------------------------------------------------------------------------- keywords and tips
# Words found in almost every job description. They say nothing about skills, so they are skipped when picking keywords.
FILLER_WORDS = set("""experience years year work works working skills skill ability abilities strong looking required requirements preferred
candidate role job position including knowledge must will etc plus need needs team good great excellent responsible responsibilities
build builds building support supports using use used able well new help based within across also such like more other join company""".split())
# Common grammar words, ignored for the same reason.
STOP_WORDS = set("""a an the and or of to in for on with at by from as is are be this that these those it its you your we our their they them who which
what when where how than then into over under about can could should would may might not no yes if else""".split())
# Weak phrases that the resume tips suggest replacing with action verbs.
VAGUE_PHRASES = ["responsible for", "worked on", "helped with", "duties included", "assisted with", "tasked with"]


def extract_job_keywords(job_description, limit=25):
    """The important words of a job description (no filler words), in the order they appear."""
    seen_words, keywords = set(), []
    for word in re.findall(r"[a-zA-Z][a-zA-Z+#.\-]{2,}", job_description.lower()):
        word = word.strip(".-")
        if len(word) > 2 and word not in STOP_WORDS and word not in FILLER_WORDS and word not in seen_words:
            seen_words.add(word)
            keywords.append(word)
    return keywords[:limit]


def keyword_overlap(job_description, resume_text):
    """Return (share of job keywords found in the resume, matched keywords, missing keywords)."""
    resume_words = set(tokenize(resume_text))
    keywords = extract_job_keywords(job_description)
    matched = [keyword for keyword in keywords if keyword in resume_words]
    missing = [keyword for keyword in keywords if keyword not in resume_words]
    return (len(matched) / len(keywords) if keywords else 0.0), matched, missing


def build_resume_tips(resume_text, sections, missing_keywords):
    """Plain advice about skills, clarity and structure. It never mentions name, gender, age, location or college."""
    tips, word_count = [], len(resume_text.split())
    if missing_keywords:
        tips.append("Add these job keywords if you really have the skill: " + ", ".join(missing_keywords[:8]) + ".")
    numbers_found = [number for number in re.findall(r"\b\d+(?:\.\d+)?%?", resume_text) if not re.fullmatch(r"(?:19|20)\d{2}", number)]
    if len(numbers_found) < 3:
        tips.append("Add numbers to your results, for example 'cut report time by 30%' or 'managed a team of 5'.")
    if "SKILLS" not in sections:
        tips.append("Add a clear Skills section so a reader (and software) can find your tools quickly.")
    if "SUMMARY" not in sections:
        tips.append("Add a 2 to 3 line summary at the top that names the job you want and your strongest skills.")
    vague_found = [phrase for phrase in VAGUE_PHRASES if phrase in resume_text.lower()]
    if vague_found:
        tips.append("Replace vague phrases (" + ", ".join(vague_found) + ") with action verbs such as 'built', 'led' or 'improved'.")
    if word_count < 250:
        tips.append("The resume is short. Add projects, tools used and measurable results.")
    elif word_count > 1200:
        tips.append("The resume is long. Keep it to the most relevant 1 to 2 pages.")
    return tips or ["Looks solid. Keep each bullet short and start it with an action verb."]


# ------------------------------------------------------------------------------------------- the engine
TWIN_A = ("Priya Nair", "She", 2003)      # made-up identities used only by the Twin Test (name, pronoun, graduation year)
TWIN_B = ("Rahul Nair", "He", 2021)


# The only class main.py uses. It loads every model once, and analyze() runs the five steps on a resume.
class FairForge:
    """Loads every trained model once and offers analyze() to the app."""

    def __init__(self):
        # Stop early with a clear message when the trained files are missing.
        if not (MODEL_DIR / "meta.json").exists():
            raise FileNotFoundError(f"Trained model files not found in {MODEL_DIR}. Unzip fairforge_models.zip into a 'models' folder next to main.py.")
        # Training settings and the list of the 24 job families, saved next to the models.
        self.meta = json.load(open(MODEL_DIR / "meta.json"))
        self.categories = self.meta["cats"]                            # the 24 job families
        self.vocab = json.load(open(MODEL_DIR / "vocab.json"))
        self.max_lines, self.max_words, self.line_length = self.meta["max_lines"], self.meta["max_words"], self.meta["line_len"]

        # Rebuild each network with the same shape it had in training, then load its saved weights.
        self.section_cnn = SectionCNN(len(self.vocab), len(SECTION_LABELS))
        self.section_cnn.load_state_dict(torch.load(MODEL_DIR / "section_cnn.pt", map_location="cpu"))
        self.career_lstm = CareerBiLSTM(len(self.vocab), len(self.categories))
        self.career_lstm.load_state_dict(torch.load(MODEL_DIR / "career_bilstm.pt", map_location="cpu"))
        self.dnn_config = json.load(open(MODEL_DIR / "fit_dnn_config.json"))
        dnn_input_size = (MINILM_VECTOR_SIZE + (len(SECTION_LABELS) if self.dnn_config["use_sections"] else 0)
                          + (CAREER_VECTOR_SIZE if self.dnn_config["use_career"] else 0))
        self.fit_dnn = FitDNN(dnn_input_size, len(self.categories))
        self.fit_dnn.load_state_dict(torch.load(MODEL_DIR / "fit_dnn.pt", map_location="cpu"))
        for network in (self.section_cnn, self.career_lstm, self.fit_dnn):
            network.eval()                                             # switch off dropout: we are predicting, not training

        self.baseline = joblib.load(MODEL_DIR / "baseline_anon.joblib")           # TF-IDF + Logistic Regression
        self.category_centroids = np.load(MODEL_DIR / "category_centroids.npy")   # average meaning vector of each job family
        self.anonymizer = Anonymizer()
        self.metrics = json.load(open(MODEL_DIR / "metrics.json")) if (MODEL_DIR / "metrics.json").exists() else {}
        # MiniLM turns text into meaning vectors. When it cannot be loaded (for example offline) the app falls back to
        # TF-IDF.
        self.encoder, self._job_vector_cache, self._shap_explainer = None, {}, None
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        except Exception:
            self.encoder = None      # offline: the app falls back to TF-IDF similarity so it never crashes
        try:
            # Files used only for the SHAP explanation. If they are missing the app just shows no explanation.
            self.explain_model = joblib.load(MODEL_DIR / "explain_lr.joblib")
            self.explain_background = np.load(MODEL_DIR / "explain_bg.npy")
        except Exception:
            self.explain_model, self.explain_background = None, None

    # ------------------------------------------------------------------ status shown in the app
    @property
    def status(self):
        """Which optional parts are working (the app shows them as small badges)."""
        return {"Name detection (spaCy)": self.anonymizer.ner_available,
                "Meaning model (MiniLM)": self.encoder is not None,
                "Explanations (SHAP)": self.explain_model is not None}

    # ------------------------------------------------------------------ neural helpers
    @torch.no_grad()
    def label_lines(self, lines):
        """CNN: one section label per line."""
        word_ids = torch.tensor(encode_to_word_ids(lines, self.vocab, self.line_length))
        predictions = [self.section_cnn(word_ids[start:start + 512]).argmax(1).numpy() for start in range(0, len(word_ids), 512)]
        return [SECTION_LABELS[index] for index in np.concatenate(predictions)]

    def build_career_input(self, lines, sections):
        """Put the first job-history lines into a [1, lines, words] matrix of word numbers for the BiLSTM."""
        experience_lines = [line for line, section in zip(lines, sections) if section == "EXPERIENCE"][:self.max_lines]
        career_matrix = np.zeros((1, self.max_lines, self.max_words), dtype=np.int64)
        for row, line in enumerate(experience_lines):
            word_ids = [self.vocab.get(word, 1) for word in tokenize(line)][:self.max_words]
            career_matrix[0, row, :len(word_ids)] = word_ids
        return career_matrix, len(experience_lines)

    @torch.no_grad()
    def career_probabilities(self, lines, sections):
        """BiLSTM: how likely each job family is, judging only by the job history. Also returns how many history lines were read."""
        career_matrix, lines_read = self.build_career_input(lines, sections)
        return softmax(self.career_lstm(torch.tensor(career_matrix))[0].numpy()), lines_read

    @torch.no_grad()
    def career_vector(self, lines, sections):
        """BiLSTM: the career vector of a resume (only needed when the DNN was trained with it)."""
        career_matrix, _ = self.build_career_input(lines, sections)
        return self.career_lstm.encode(torch.tensor(career_matrix))[0].numpy()

    def embed(self, texts):
        """MiniLM meaning vectors (None when MiniLM is not available)."""
        if self.encoder is None:
            return None
        return self.encoder.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    def job_vector(self, job_description):
        """Meaning vector of the job description (remembered, so it is computed once)."""
        if job_description not in self._job_vector_cache:
            embedding = self.embed([job_description])
            self._job_vector_cache[job_description] = None if embedding is None else embedding[0]
        return self._job_vector_cache[job_description]

    # ------------------------------------------------------------------ step 1: read the resume and hide identity
    def prepare(self, resume_text, hide_identity=True):
        """Split into lines, hide identity (optional), label sections, and build the text the models will read."""
        # 1. Read the resume as lines and hide the identity details.
        original_lines = split_into_lines(resume_text)
        hidden_counts = collections.Counter()
        if hide_identity:
            model_lines, hidden_counts = self.anonymizer.hide_lines(original_lines)
        else:
            model_lines = list(original_lines)
        # 2. Let the CNN label every line with its section.
        sections = self.label_lines(model_lines)
        if hide_identity:      # graduation years are hidden only in Education lines, so job dates stay readable
            model_lines = [self.anonymizer.hide_years(line, hidden_counts) if section == "EDUCATION" else line
                           for line, section in zip(model_lines, sections)]
        # The header (name and contacts) is left out of the text the models read. Very short resumes keep every line.
        body_text = " ".join(line for line, section in zip(model_lines, sections) if section != "HEADER")   # the header holds name and contacts
        model_text = body_text if len(body_text.split()) >= 40 else " ".join(model_lines)
        return {"lines": original_lines, "anon": model_lines, "sections": sections, "model_text": model_text, "counts": hidden_counts}

    # ------------------------------------------------------------------ step 2: score
    def score(self, resume_text, job_description, weights, hide_identity=True):
        """Score one resume against a job description. weights = how much each of the four signals counts."""
        # Steps 1 and 2: hide identity and sort the sections.
        prepared = self.prepare(resume_text, hide_identity)
        # Meaning vectors for the resume and the job description.
        resume_embedding = self.embed([prepared["model_text"]])
        job_embedding = self.job_vector(job_description)
        tfidf_vectorizer, tfidf_model = self.baseline["vectorizer"], self.baseline["model"]
        resume_vector_for_chart = None
        if resume_embedding is not None and job_embedding is not None:
            resume_embedding = resume_embedding[0]
            resume_vector_for_chart = resume_embedding
            meaning_match = float(np.clip(resume_embedding @ job_embedding, 0, 1))               # 1 = same meaning
            job_family_weights = softmax(self.category_centroids @ job_embedding / 0.05)         # which job family does the description look like?
            # Build the DNN input: the meaning vector, plus section shares and the career vector when the DNN was
            # trained with them.
            feature_blocks = [resume_embedding]
            if self.dnn_config["use_sections"]:
                total_lines = max(len(prepared["sections"]), 1)
                feature_blocks.append(np.array([prepared["sections"].count(label) / total_lines for label in SECTION_LABELS], dtype=np.float32))
            if self.dnn_config["use_career"]:
                feature_blocks.append(self.career_vector(prepared["anon"], prepared["sections"]))
            with torch.no_grad():
                dnn_input = torch.tensor(np.concatenate(feature_blocks).astype(np.float32)[None])
                job_fit_probabilities = softmax(self.fit_dnn(dnn_input)[0].numpy())
            engine_name = "MiniLM + DNN"
        else:                                                                    # offline fallback: TF-IDF instead of MiniLM
            # Fallback when MiniLM is not available: the same signals, computed with TF-IDF instead.
            resume_tfidf, job_tfidf = tfidf_vectorizer.transform([prepared["model_text"]]), tfidf_vectorizer.transform([job_description])
            length_product = (np.sqrt(resume_tfidf.multiply(resume_tfidf).sum()) * np.sqrt(job_tfidf.multiply(job_tfidf).sum())) or 1.0
            meaning_match = float(resume_tfidf.multiply(job_tfidf).sum() / length_product)
            job_family_weights = tfidf_model.predict_proba(job_tfidf)[0]
            job_fit_probabilities = tfidf_model.predict_proba(resume_tfidf)[0]
            engine_name = "TF-IDF fallback (MiniLM not available)"
        # Step 3: the BiLSTM reads only the job-history lines.
        career_probabilities, history_lines_read = self.career_probabilities(prepared["anon"], prepared["sections"])
        # Signal 4: how many of the job's key words appear in the resume.
        keyword_share, matched_keywords, missing_keywords = keyword_overlap(job_description, prepared["model_text"])
        # The four signals, each between 0 and 1. Job fit and career path are averaged using job_family_weights, which
        # says how much the job description resembles each job family.
        signal_scores = {"Meaning match": meaning_match,
                         "Job fit (DNN)": float(job_family_weights @ job_fit_probabilities),
                         "Career path (BiLSTM)": float(job_family_weights @ career_probabilities),
                         "Skill keywords": keyword_share}
        # Step 5: weighted average of the four signals, using the slider weights from the interface.
        total_weight = sum(weights.values()) or 1.0
        final_score = sum(weights[name] * signal_scores[name] for name in signal_scores) / total_weight
        # The three job families the description looks most like (shown in the app).
        top_families = np.argsort(-job_family_weights)[:3]
        return {"final": float(final_score), "parts": signal_scores, "family": job_family_weights,
                "top_family": [(self.categories[index], float(job_family_weights[index])) for index in top_families],
                "matched": matched_keywords, "missing": missing_keywords, "engine": engine_name, "experience_lines": history_lines_read,
                "emb": resume_vector_for_chart, "emb_jd": job_embedding, **prepared}

    # ------------------------------------------------------------------ step 3: explain with SHAP
    def explain(self, model_text, job_family_weights, words_up=8, words_down=5):
        """SHAP: which words pushed the match towards (up) or away from (down) the most likely job family."""
        if self.explain_model is None:
            return None
        try:
            import shap
            # SHAP explains the word-level model (explain_lr.joblib), not the neural networks.
            word_vectorizer, word_classifier = self.explain_model["vectorizer"], self.explain_model["model"]
            if self._shap_explainer is None:
                self._shap_explainer = shap.LinearExplainer(word_classifier, shap.maskers.Independent(self.explain_background, max_samples=len(self.explain_background)))
            word_features = word_vectorizer.transform([model_text]).toarray().astype(np.float32)
            shap_output = self._shap_explainer.shap_values(word_features)
            shap_values = np.stack(shap_output, axis=-1) if isinstance(shap_output, list) else np.asarray(shap_output)
            if shap_values.ndim == 3 and shap_values.shape[0] == len(self.categories) and shap_values.shape[2] != len(self.categories):
                shap_values = np.transpose(shap_values, (1, 2, 0))              # make the shape [samples, words, job families]
            # Explain the job family the description matches best: which words pushed towards it and which pushed away.
            family_index = int(np.argmax(job_family_weights))
            word_effects, vocabulary_words = shap_values[0, :, family_index], word_vectorizer.get_feature_names_out()
            strongest_up, strongest_down = np.argsort(-word_effects)[:words_up], np.argsort(word_effects)[:words_down]
            return {"family": self.categories[family_index],
                    "up": [(vocabulary_words[i], float(word_effects[i])) for i in strongest_up if word_effects[i] > 0],
                    "down": [(vocabulary_words[i], float(word_effects[i])) for i in strongest_down if word_effects[i] < 0]}
        except Exception:
            return None

    # ------------------------------------------------------------------ step 4: the Twin Test
    def twin_test(self, resume_text, job_description, weights):
        """Same resume, two made-up identities (different name, pronoun and graduation year). The score must not move."""
        outcome = {}
        # Score the resume twice, once with identity hidden and once visible. Each time it is given two made-up
        # identities (Twin A and Twin B) and the two scores are compared.
        for hide_identity, label in ((True, "identity hidden"), (False, "identity visible")):
            twin_scores = [self.score(f"{name}\n{pronoun} graduated in {year}.\n{resume_text}", job_description, weights, hide_identity)["final"]
                           for name, pronoun, year in (TWIN_A, TWIN_B)]
            outcome[label] = {"twin A": twin_scores[0], "twin B": twin_scores[1], "gap": abs(twin_scores[0] - twin_scores[1])}
        outcome["pass"] = outcome["identity hidden"]["gap"] < 0.005       # under half a score point
        return outcome

    # ------------------------------------------------------------------ everything the app needs for one resume
    def analyze(self, resume_text, job_description, weights, run_twin_test=True):
        """Full analysis of one resume: scores, explanation, tips and (optionally) the Twin Test."""
        # The official score always uses the identity-hidden resume.
        hidden_result = self.score(resume_text, job_description, weights, hide_identity=True)
        visible_result = self.score(resume_text, job_description, weights, hide_identity=False)      # only used to compare ranks
        return {"hidden": hidden_result, "visible_score": visible_result["final"],
                "explain": self.explain(hidden_result["model_text"], hidden_result["family"]),
                "tips": build_resume_tips(hidden_result["model_text"], hidden_result["sections"], hidden_result["missing"]),
                "twin": self.twin_test(resume_text, job_description, weights) if run_twin_test else None, "raw_text": resume_text}
