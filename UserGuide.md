# FairForge: User Manual

FairForge ranks resumes against a job description **without looking at who the candidate is**. It hides names, places, colleges, pronouns, contact details and graduation years before any model reads the resume, explains every score, and tests itself for fairness.

> FairForge is decision support, not a decision maker. It hides identity details and explains its scores, but it cannot promise zero bias. A person must make the final hiring decision. All identities used in the fairness tests are synthetic (made up).

---

## 1. Quick start (2 minutes)

1. Open the FairForge website.
2. Leave the job description as it is and keep **Use the 3 demo resumes** ticked.
3. Press **Ignite the forge**.
4. Use the tab bar at the bottom of the screen to look at the results.

That is the whole flow. The rest of this manual explains each part.

---

## 2. The page, from top to bottom

### Hero banner
An animated banner at the top. It tells the story of the app in four phases: *Twins, Blind, Skills, Fair*. Click a phase name to jump to it. It is only decoration and does not affect any score.

### Forge console

| Part | What to do |
|---|---|
| **Step 1: Describe the job** | Paste the job description you are hiring for. A sample (Data Analyst) is already filled in. |
| **Step 2: Drop the resumes** | Upload one or more resumes as **PDF** or **TXT** files. You can select many files at once. |
| **Use the 3 demo resumes** | Ticked automatically when nothing is uploaded. Tick it to add the demo resumes to your own files, or untick it to use only your uploads. |

### Mix the score
Four sliders decide how much each signal counts in the final score. The coloured bar below them shows the mix live.

| Slider | What it measures | Default |
|---|---|---|
| **Meaning match (MiniLM)** | Does the resume mean the same thing as the job description? | 30 |
| **Job fit (DNN)** | How likely is the resume to belong to the job's family of roles? | 30 |
| **Career path (BiLSTM)** | Does the order of past jobs fit the role? | 15 |
| **Skill keywords** | How many of the job's key skills appear in the resume? | 25 |

The numbers are relative. They do not have to add up to 100. If every slider is set to 0, the app uses skill keywords only.

### Status lights
Small coloured dots show which parts of the system are running:

- **Teal** means the part is working.
- **Amber** means a simpler fallback is being used. For example, if the meaning model (MiniLM) is not available, the app switches to a basic TF-IDF comparison and still gives a score. Name hiding is also less reliable when *Name detection (spaCy)* is amber.

### Twin Test switch and the Ignite button
- **Run the Twin Test (slower)**: on by default. It scores every resume several more times to check fairness (see section 4). Turn it off if you want faster results.
- **Ignite the forge**: starts the ranking. Nothing is scored until you press it. Press it again after changing the job description, the files or the sliders.

---

## 3. Reading the results

Once ranking is done, a bar floats at the bottom of the screen. Click a tab to switch views.

### Ranking
- **Podium**: the top three candidates with their score out of 100.
- **Leaderboard**: every candidate with a score bar.
- **Rank movement**: each candidate shows how their rank changed once identity was hidden (▲ up, ▼ down, ● same). This shows whether identity would have changed the order.
- **Radar chart**: the shape of each candidate's four signal scores.
- **Slope chart**: rank before and after hiding identity.
- **Score table**: all scores side by side. *Rank change* is positive when a candidate moved up after identity was hidden.
- **Job-fit galaxy**: a 3D chart you can drag. The closer a candidate sits to the job description, the closer the meaning of the resume.
- **Key**: to keep the review blind, candidates are shown as *Candidate 1, 2, 3...*. Open the **Key** box at the bottom to see which file is which.

### Why this rank
Pick a candidate to see:
- The final score and the four signal scores.
- The job families that the job description looks most like.
- A **word-by-word chart (SHAP)**: teal words raised the match, orange words lowered it.
- **Skill gap**: green chips are job keywords found in the resume, orange chips are missing ones.
- **Resume tips**: advice on skills, clarity and structure. The tips never mention identity.

### Hidden identity
Pick a candidate to compare two versions of the same resume:
- **Left**: what a person would see.
- **Right**: what the scoring models see. Black bars show details that were replaced (name, place, email, phone, college, pronoun, year). The coloured label on each line is the section found by the section model (Summary, Skills, Experience, Education...).
- Two charts show what kinds of details were hidden and how the resume splits into sections.

### Fairness audit
- **Twin Test**: see section 4.
- **Bias audit from training**: numbers measured while the models were trained, on made-up identities. It shows disparate impact, how well an adversary model can still guess gender or age, how many injected identity details the anonymizer caught, and the accuracy of the job-category models.

### About
A short overview of the five scoring steps and the known limits.

---

## 4. What the Twin Test means

The Twin Test takes your candidate's resume and gives it two made-up identities:

- **Twin A**: Priya Nair, She, graduated 2003
- **Twin B**: Rahul Nair, He, graduated 2021

It scores both twins, once with identity visible and once with identity hidden. The resume content is exactly the same, so the two scores should be the same.

- Each balance scale tilts with the score gap. The tilt is exaggerated so it is easy to see: every point of gap turns the beam 4 degrees.
- **PASS**: the gap with identity hidden is under 0.5 score points.
- **CHECK**: the gap is 0.5 points or more. Treat that score with extra care.

---

## 5. How a resume is scored

1. **Hide identity**: names, places, colleges, pronouns, contact details and graduation years are removed.
2. **Sort sections**: every line is labelled as Summary, Skills, Experience, Education or Other.
3. **Read the career**: the job-history lines are read in order, forwards and backwards.
4. **Judge the fit**: the resume is turned into meaning numbers and a job-fit probability is calculated.
5. **Score and check**: the four signals are mixed using your slider weights, SHAP explains the result, and the Twin Test checks fairness.

---

## 6. Tips for good results

- Use resumes with real text. A scanned PDF (a picture of a page) has no text to read and will be skipped. Convert it with OCR first.
- A resume with fewer than 30 words is skipped with a warning. The other resumes are still ranked.
- Write the job description in full sentences and name the tools and skills you need. Both the meaning match and the keyword signal depend on it.
- Try different slider mixes. If two mixes give the same top candidate, the result is stable.
- Compare the **Hidden identity** tab with the **Ranking** tab if a rank movement looks surprising.

---

## 7. Troubleshooting

| What you see | What to do |
|---|---|
| "Skipped ...: almost no text found" | The file is probably a scanned PDF. Run OCR on it first, or upload a TXT version. |
| An amber status light | The app is using a fallback and still works. Scores may differ slightly from the full version. |
| Results look old after changing something | Press **Ignite the forge** again. |
| No Twin Test results in the Fairness audit tab | Switch on **Run the Twin Test** and rank again. |
| No word-by-word (SHAP) chart for a candidate | The explanation is not available for that resume. The score itself is not affected. |
| Plain fonts or a simple banner instead of the animation | These load from the internet. Without a connection everything else still works. |
