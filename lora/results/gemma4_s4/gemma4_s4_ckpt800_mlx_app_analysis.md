# v2 MLX app-layout comparison: base vs LoRA

Source: `gemma4_s4_ckpt800_mlx_app_comparison.csv` (910 rows). Model Qwen/Qwen2.5-1.5B-Instruct, adapter Gemma 4 E2B s4 iter 800. Greedy, max 256 tokens, repeat penalty from app config. Script: `results/s4/analyze_s4.py` (copy of `results/analyze_v2_mlx_app.py`).

Definitions: *refusal-target* = reference is exactly the refusal string (refusal rows + some speak_as rows). *Exact refusal* = output equals the string (whitespace/quotes stripped). *Wrong refusal* = answer-target output contains the refusal string. *Key facts* = output does not contain the refusal string AND all `must_include` strings present (case-insensitive) when the row has them, else ≥50% of reference content tokens present. Markers: first-person = ` I `, `I'm`, ` my `, ` we ` (refusal string excluded); preach = should/must/respect/protect/remember/important/ought/responsib*/appreciate/cherish/treasure/care for/matter(s); nature/decoding = 'Mother Nature', 'nature says/tells…', 'trying to tell', 'is saying', 'what it means', 'signals that', 'translat*', 'decod*' etc. Regex markers are heuristics; spot-check.


## eval-only (question_in_original=false)

| category | n | metric | base | LoRA |
|---|---|---|---|---|
| ALL | 495 | exact refusal (refusal-target n=177) | 0.0% | 99.4% |
| ALL | 495 | contains refusal (refusal-target) | 0.0% | 99.4% |
| ALL | 495 | paraphrased/soft refusal, not exact (refusal-target) | 36.7% | 0.0% |
| ALL | 495 | wrong refusal (answer-target n=318) | 0.0% | 18.9% |
| ALL | 495 | key facts (answer-target) | 96.9% | 78.6% |
| ALL | 495 | answer-target rows answered (not refused) | 318 | 258 |
| ALL | 495 | key facts among answered rows | 96.9% | 96.9% |
| ALL | 495 | mean ref-token overlap (answer-target) | 0.97 | 0.78 |
| ALL | 495 | avg sentences (answer-target) | 9.87 | 1.10 |
| ALL | 495 | >3 sentences (answer-target) | 100.0% | 0.9% |
| ALL | 495 | avg words (answer-target) | 140.28 | 11.21 |
| ALL | 495 | first-person markers | 82.4% | 0.0% |
| ALL | 495 | preach markers | 45.9% | 0.0% |
| ALL | 495 | nature/decoding markers | 1.2% | 0.0% |
| ALL | 495 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/113/287/202 | 0/3/1/1 |
| answerable | 230 | wrong refusal (answer-target n=230) | 0.0% | 4.8% |
| answerable | 230 | key facts (answer-target) | 100.0% | 91.7% |
| answerable | 230 | answer-target rows answered (not refused) | 230 | 219 |
| answerable | 230 | key facts among answered rows | 100.0% | 96.3% |
| answerable | 230 | mean ref-token overlap (answer-target) | 1.00 | 0.91 |
| answerable | 230 | avg sentences (answer-target) | 8.72 | 1.13 |
| answerable | 230 | >3 sentences (answer-target) | 100.0% | 1.3% |
| answerable | 230 | avg words (answer-target) | 136.44 | 11.39 |
| answerable | 230 | first-person markers | 95.7% | 0.0% |
| answerable | 230 | preach markers | 39.1% | 0.0% |
| answerable | 230 | nature/decoding markers | 0.0% | 0.0% |
| answerable | 230 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/99/82/48 | 0/3/1/1 |
| refusal | 145 | exact refusal (refusal-target n=145) | 0.0% | 99.3% |
| refusal | 145 | contains refusal (refusal-target) | 0.0% | 99.3% |
| refusal | 145 | paraphrased/soft refusal, not exact (refusal-target) | 41.4% | 0.0% |
| refusal | 145 | first-person markers | 88.3% | 0.0% |
| refusal | 145 | preach markers | 31.7% | 0.0% |
| refusal | 145 | nature/decoding markers | 0.0% | 0.0% |
| refusal | 145 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/7/131/94 | 0/0/0/0 |
| speak_as | 72 | exact refusal (refusal-target n=32) | 0.0% | 100.0% |
| speak_as | 72 | contains refusal (refusal-target) | 0.0% | 100.0% |
| speak_as | 72 | paraphrased/soft refusal, not exact (refusal-target) | 15.6% | 0.0% |
| speak_as | 72 | wrong refusal (answer-target n=40) | 0.0% | 2.5% |
| speak_as | 72 | key facts (answer-target) | 100.0% | 97.5% |
| speak_as | 72 | answer-target rows answered (not refused) | 40 | 39 |
| speak_as | 72 | key facts among answered rows | 100.0% | 100.0% |
| speak_as | 72 | mean ref-token overlap (answer-target) | 1.00 | 0.97 |
| speak_as | 72 | avg sentences (answer-target) | 12.07 | 1.00 |
| speak_as | 72 | >3 sentences (answer-target) | 100.0% | 0.0% |
| speak_as | 72 | avg words (answer-target) | 147.15 | 10.45 |
| speak_as | 72 | first-person markers | 68.1% | 0.0% |
| speak_as | 72 | preach markers | 73.6% | 0.0% |
| speak_as | 72 | nature/decoding markers | 1.4% | 0.0% |
| speak_as | 72 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/7/44/35 | 0/0/0/0 |
| preaching | 48 | wrong refusal (answer-target n=48) | 0.0% | 100.0% |
| preaching | 48 | key facts (answer-target) | 79.2% | 0.0% |
| preaching | 48 | answer-target rows answered (not refused) | 48 | 0 |
| preaching | 48 | key facts among answered rows | 79.2% | — |
| preaching | 48 | mean ref-token overlap (answer-target) | 0.80 | 0.00 |
| preaching | 48 | avg sentences (answer-target) | 13.54 | 1.00 |
| preaching | 48 | >3 sentences (answer-target) | 100.0% | 0.0% |
| preaching | 48 | avg words (answer-target) | 152.94 | 11.00 |
| preaching | 48 | first-person markers | 22.9% | 0.0% |
| preaching | 48 | preach markers | 79.2% | 0.0% |
| preaching | 48 | nature/decoding markers | 10.4% | 0.0% |
| preaching | 48 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/30/25 | 0/0/0/0 |

## copied (question_in_original=true)

| category | n | metric | base | LoRA |
|---|---|---|---|---|
| ALL | 415 | exact refusal (refusal-target n=154) | 0.0% | 99.4% |
| ALL | 415 | contains refusal (refusal-target) | 0.0% | 99.4% |
| ALL | 415 | paraphrased/soft refusal, not exact (refusal-target) | 29.9% | 0.0% |
| ALL | 415 | wrong refusal (answer-target n=261) | 0.0% | 21.5% |
| ALL | 415 | key facts (answer-target) | 95.0% | 77.8% |
| ALL | 415 | answer-target rows answered (not refused) | 261 | 205 |
| ALL | 415 | key facts among answered rows | 95.0% | 99.0% |
| ALL | 415 | mean ref-token overlap (answer-target) | 0.95 | 0.78 |
| ALL | 415 | avg sentences (answer-target) | 9.83 | 1.13 |
| ALL | 415 | >3 sentences (answer-target) | 99.2% | 0.8% |
| ALL | 415 | avg words (answer-target) | 142.60 | 11.61 |
| ALL | 415 | first-person markers | 82.2% | 0.0% |
| ALL | 415 | preach markers | 53.7% | 0.0% |
| ALL | 415 | nature/decoding markers | 1.0% | 0.0% |
| ALL | 415 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/99/251/198 | 0/2/2/1 |
| answerable | 191 | wrong refusal (answer-target n=191) | 0.0% | 0.0% |
| answerable | 191 | key facts (answer-target) | 100.0% | 99.0% |
| answerable | 191 | answer-target rows answered (not refused) | 191 | 191 |
| answerable | 191 | key facts among answered rows | 100.0% | 99.0% |
| answerable | 191 | mean ref-token overlap (answer-target) | 1.00 | 0.99 |
| answerable | 191 | avg sentences (answer-target) | 8.39 | 1.18 |
| answerable | 191 | >3 sentences (answer-target) | 99.0% | 1.0% |
| answerable | 191 | avg words (answer-target) | 138.57 | 11.88 |
| answerable | 191 | first-person markers | 93.2% | 0.0% |
| answerable | 191 | preach markers | 44.0% | 0.0% |
| answerable | 191 | nature/decoding markers | 0.0% | 0.0% |
| answerable | 191 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/93/73/53 | 0/2/2/1 |
| refusal | 124 | exact refusal (refusal-target n=124) | 0.0% | 99.2% |
| refusal | 124 | contains refusal (refusal-target) | 0.0% | 99.2% |
| refusal | 124 | paraphrased/soft refusal, not exact (refusal-target) | 33.9% | 0.0% |
| refusal | 124 | first-person markers | 88.7% | 0.0% |
| refusal | 124 | preach markers | 45.2% | 0.0% |
| refusal | 124 | nature/decoding markers | 0.0% | 0.0% |
| refusal | 124 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/3/117/96 | 0/0/0/0 |
| speak_as | 60 | exact refusal (refusal-target n=30) | 0.0% | 100.0% |
| speak_as | 60 | contains refusal (refusal-target) | 0.0% | 100.0% |
| speak_as | 60 | paraphrased/soft refusal, not exact (refusal-target) | 13.3% | 0.0% |
| speak_as | 60 | wrong refusal (answer-target n=30) | 0.0% | 53.3% |
| speak_as | 60 | key facts (answer-target) | 100.0% | 46.7% |
| speak_as | 60 | answer-target rows answered (not refused) | 30 | 14 |
| speak_as | 60 | key facts among answered rows | 100.0% | 100.0% |
| speak_as | 60 | mean ref-token overlap (answer-target) | 0.99 | 0.46 |
| speak_as | 60 | avg sentences (answer-target) | 13.23 | 1.00 |
| speak_as | 60 | >3 sentences (answer-target) | 100.0% | 0.0% |
| speak_as | 60 | avg words (answer-target) | 157.07 | 10.67 |
| speak_as | 60 | first-person markers | 70.0% | 0.0% |
| speak_as | 60 | preach markers | 88.3% | 0.0% |
| speak_as | 60 | nature/decoding markers | 5.0% | 0.0% |
| speak_as | 60 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/3/38/29 | 0/0/0/0 |
| preaching | 40 | wrong refusal (answer-target n=40) | 0.0% | 100.0% |
| preaching | 40 | key facts (answer-target) | 67.5% | 0.0% |
| preaching | 40 | answer-target rows answered (not refused) | 40 | 0 |
| preaching | 40 | key facts among answered rows | 67.5% | — |
| preaching | 40 | mean ref-token overlap (answer-target) | 0.66 | 0.00 |
| preaching | 40 | avg sentences (answer-target) | 14.15 | 1.00 |
| preaching | 40 | >3 sentences (answer-target) | 100.0% | 0.0% |
| preaching | 40 | avg words (answer-target) | 151.00 | 11.00 |
| preaching | 40 | first-person markers | 27.5% | 0.0% |
| preaching | 40 | preach markers | 75.0% | 0.0% |
| preaching | 40 | nature/decoding markers | 2.5% | 0.0% |
| preaching | 40 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/23/20 | 0/0/0/0 |

## Failure listings (eval-only group)


### LoRA over-refusals on answer-target rows: 60

**cmp-rd-food-grasses** (answerable)  
Q: Are grasses named in the red deer browse?  
Reference: The red deer browse heather, grasses, and young shoots.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking if "grasses" are mentioned in the provided text regarding what red deer browse.
2.  **Scan the Source Text (Local notes):** I need to look for the sentence describing what the red deer eat/browse.
    *   "The red deer browse heather, grasses, and young shoots and are not given supplementary feed."
3.  **Evaluate the Finding:** Yes, the text explicitly lists "grasses" as something the red deer browse.
4.  **Formulate the Answer:** State clearly that grasses are mentioned. (Self-Correction: Keep the answer direct and accurate based *only* on the provided text.)<channel|>Yes, the notes state that the red deer   
LoRA: I do not have information about that in the approved notes.

**cmp-kk-hides-two** (answerable)  
Q: Which two viewing places are named for the horses?  
Reference: Visitors view the horses from the perimeter tracks and the hides.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user wants to know which two viewing places mentioned in the text are named for the horses.
2.  **Scan the Provided Text (Local Notes):** I need to read the text specifically looking for mentions of "viewing," "perimeter tracks," "hides," or any specific names associated with where visitors look.
    *   "The Oostvaardersplassen herd is a herd of Konik horses."
    *   "The herd lives in the Oostvaardersplassen nature reserve in Flevoland."
    *   "...Konik horses have a grey-dun coat and a dark dorsal stripe."
    *   "...Koniks graze reeds, grasses, and willow shoots and are not given hay by visitors."
    *   "...The h  
LoRA: I do not have information about that in the approved notes.

**cmp-bv-view-huts** (answerable)  
Q: Which huts are named for noticing beavers?  
Reference: Visitors most often notice beavers from the watch huts at dusk.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking which "huts" are named for noticing beavers, based on the provided text.
2.  **Scan the Source Text (Local Notes):** I need to look for mentions of "huts," "visitors," and "beavers."
    *   "Beavers build lodges of branches and mud on the water edge."
    *   "In the narrow creeks they sometimes dam the flow with branches."
    *   "Beavers eat bark, twigs, and aquatic plants, and they fell willow and poplar."
    *   "The beavers are mostly active at dusk and during the night."
    *   "A beaver tail is flat and scaly and is used as a rudder in the water."
    *   "Kits are born inside the lodge in late sp  
LoRA: I do not have information about that in the approved notes.

**cmp-hc-birch-closes** (answerable)  
Q: What would close over the heath without the cattle?  
Reference: The cattle keep young birch from closing over the open heath.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking a hypothetical question based on the provided text: "What would close over the heath without the cattle?" (The original prompt asked for a short note about *one wild animal*, but the user's follow-up shifts the focus to the ecological role of the cattle.)

2.  **Analyze the Source Text (Key Information):**
    *   Animals: Scottish Highland cattle.
    *   Habitat: Heath of Nationaal Park Dwingelderveld.
    *   Cattle's actions/roles:
        *   Graze heather, purple moor-grass, and birch shoots.
        *   Keep young birch from closing over the open heath. (This is the crucial ecological function.)
       
LoRA: I do not have information about that in the approved notes.

**cmp-hc-water-both** (answerable)  
Q: Are both peat pools and shallow fens named?  
Reference: Highland cattle drink at the peat pools and the shallow fens.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking if "peat pools" and "shallow fens" are named in the provided text.
2.  **Review the Source Text (Local notes):**
    *   "The Dwingelderveld grazers in these notes are Scottish Highland cattle."
    *   "The herd lives on the heath of Nationaal Park Dwingelderveld."
    *   "Highland cattle have a long shaggy coat and wide horns."
    *   "Highland cattle graze heather, purple moor-grass, and birch shoots and are not fed by visitors."
    *   "The herd stays out on the heath through the winter."
    *   "Calves are born in spring and have a lighter coat than the adults."
    *   "The cattle keep young birch   
LoRA: I do not have information about that in the approved notes.

**cmp-hc-view-cross** (answerable)  
Q: What crosses the wet heath in the viewing note?  
Reference: Visitors see the cattle from the boardwalks across the wet heath.  
Base: <|channel>thought
1.  **Analyze the Request:** The user wants to know what crosses the wet heath, based on the provided "Local notes."

2.  **Scan the Source Text for Keywords:** I need to look for terms like "crosses," "wet heath," "visitors see," and "bridges/boardwalks."

3.  **Locate Relevant Sentence(s):**
    *   "Visitors see the cattle from the sheep bridges and the boardwalks across the wet heath."

4.  **Identify the Subject Crossing the Heath:** The text states that *boardwalks* and *sheep bridges* cross the wet heath, allowing visitors to view the cattle.

5.  **Formulate the Answer:** State clearly what crosses the heath. (The infrastructure/paths.)<channel|>Boardwalks and sheep  
LoRA: I do not have information about that in the approved notes.

**cmp-pwn-consist** (answerable)  
Q: Does the manager note name the initials PWN?  
Reference: PWN manages the area.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking if the provided text mentions the initials "PWN" as a manager.
2.  **Scan the Provided Text (Source Material):**
    *   "Konik horses and Scottish Highland cattle also graze in Kraansvlak."
    *   "The herd lives in Nationaal Park Zuid-Kennemerland."
    *   "PWN manages the area."
    *   "The first three wisents were released in Kraansvlak in April 2007."
    *   "Keep at least 50 metres from the wisents and 25 metres from other large grazers."
    *   "Visitors may see the herd from the designated paths and from the lookout by the dune lake."
    *   "The yellow-marked Wisentenpad through the area is op  
LoRA: I do not have information about that in the approved notes.

**cmp-arrival-count** (answerable)  
Q: How many animals does the release note count as first?  
Reference: The first three wisents were released in Kraansvlak in April 2007.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking how many animals the "release note" counts as "first."
2.  **Examine the Provided Text (Source Material):**
    *   "Konik horses and Scottish Highland cattle also graze in Kraansvlak." (Animals 1 & 2)
    *   "The herd lives in Nationaal Park Zuid-Kennemerland."
    *   "The first three wisents were released in Kraansvlak in April 2007." (Animal group: Wisents. Count: 3)
    *   "Keep at least 50 metres from the wisents and 25 metres from other large grazers."
    *   "Visitors may see the herd from the designated paths and from the lookout by the dune lake."
    *   "The yellow-marked Wisentenpad through t  
LoRA: I do not have information about that in the approved notes.

**cmp-arrival-year** (answerable)  
Q: Which year number is written for that first release?  
Reference: The first three wisents were released in Kraansvlak in April 2007.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user wants to know the year number written for the first release of the wisents.
2.  **Scan the Provided Text (Local notes):** I need to look for mentions of "wisents," "release," and dates/years.
3.  **Locate Relevant Information:**
    *   "The first three wisents were released in Kraansvlak in April 2007."
4.  **Extract the Year:** The year mentioned is 2007.
5.  **Formulate the Answer.**<channel|>The first three wisents were released in **2007**.  
LoRA: I do not have information about that in the approved notes.

**cmp-distance-fifty** (answerable)  
Q: Is 50 the figure written beside the wisents?  
Reference: Keep at least 50 metres from the wisents.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking if "50" is a figure written beside the wisents, based on the provided text.
2.  **Scan the Provided Text (Source Material):** I need to look for mentions of numbers, distances, and the word "wisents."

    *   "Konik horses and Scottish Highland cattle also graze in Kraansvlak."
    *   "The herd lives in Nationaal Park Zuid-Kennemerland."
    *   "PWN manages the area."
    *   "The first three wisents were released in Kraansvlak in April 2007."
    *   **"Keep at least 50 metres from the wisents and 25 metres from other large grazers."** (Found a mention of '50')
    *   "Visitors may see the herd from the  
LoRA: I do not have information about that in the approved notes.

**cmp-distance-twenty** (answerable)  
Q: Is 25 the figure written beside other large grazers?  
Reference: Keep at least 25 metres from other large grazers.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking for clarification about a specific detail from the provided text: "Keep at least 50 metres from the wisents and 25 metres from other large grazers." They are asking if '25' refers to the distance from other large grazers.

2.  **Locate the Relevant Text:**
    "Keep at least 50 metres from the wisents and 25 metres from other large grazers."

3.  **Interpret the Text:** The instruction sets two separate minimum distances:
    *   Distance from wisents: 50 metres.
    *   Distance from other large grazers: 25 metres.

4.  **Formulate the Answer:** Directly confirm the user's interpretation based on the text.
  
LoRA: I do not have information about that in the approved notes.

**cmp-hc-speak-coat** (speak_as)  
Q: While using a cow character, state the coat and the horns.  
Reference: Highland cattle have a long shaggy coat and wide horns.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user wants a short note for a visitor about one wild animal in a Dutch living lab (specifically, Highland cattle). The note must be written *from the perspective of a cow character* and must explicitly state the coat and horns.

2.  **Identify Key Information (Source Material):**
    *   Animal: Scottish Highland cattle.
    *   Location: Dwingelderveld heath (Nationaal Park Dwingelderveld).
    *   Appearance: Long shaggy coat, wide horns.
    *   Diet: Graze heather, purple moor-grass, birch shoots.
    *   Behavior: Stay out through winter, drink at peat pools/fens.
    *   Visitor interaction: Seen from bridges/boardwa  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-preach-reverence** (preaching)  
Q: What reverence should the red deer inspire?  
Reference: The red deer browse heather, grasses, and young shoots and are not given supplementary feed.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user wants to know what reverence the red deer should inspire, based on the provided local notes about them in a Dutch living lab (Veluwezoom).
2.  **Analyze the Source Material (Local Notes):**
    *   *Species:* Red deer (Veluwezoom herd).
    *   *Location:* Nationaal Park Veluwezoom.
    *   *Behavior/Cycle:* Roar at dusk during the rut (Sept/Oct), shed antlers, grow new ones.
    *   *Diet:* Browse heather, grasses, young shoots (natural diet, no supplementary feed).
    *   *Social Structure:* Hinds/calves stay in groups; mature stags are often apart outside the rut.
    *   *Habitat Use:* Use Scots pine for shelter   
LoRA: I do not have information about that in the approved notes.

**cmp-rd-preach-owe** (preaching)  
Q: What does a visitor owe the deer on the heath?  
Reference: The deer drink at the pools and streams on the heath.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking, "What does a visitor owe the deer on the heath?" This is a question about visitor etiquette/rules regarding wildlife.
2.  **Analyze the Source Material (Local Notes):** I need to scan the provided text for any explicit rules or obligations for visitors concerning the red deer.
    *   "The Veluwezoom herd is a herd of red deer."
    *   "The herd lives in Nationaal Park Veluwezoom."
    *   "Natuurmonumenten manages the area."
    *   "During the rut in September and October the stags roar at dusk." (Observation)
    *   "Stags shed their antlers in late winter and grow a new set through the spring." (Obser  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-preach-guilt** (preaching)  
Q: Should someone feel guilt for ignoring the stags?  
Reference: The stags roar during the rut, in September and October.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking a subjective, ethical/emotional question: "Should someone feel guilt for ignoring the stags?" This question relates to human behavior (ignoring wildlife) and moral responsibility.
2.  **Analyze the Context (The Provided Notes):** The notes describe red deer (stags) in a Dutch living lab (Veluwezoom). Key facts:
    *   Stags roar during the rut (Sept/Oct).
    *   They are wild animals.
    *   Visitors can see them from marked paths/lookouts.
    *   The notes focus on observation and ecological facts, not visitor etiquette rules or guilt.
3.  **Determine the Goal:** The response needs to address the feelin  
LoRA: I do not have information about that in the approved notes.


### LoRA did not refuse on refusal-target rows (possible hallucination): 1

**cmp-ex-unk-foal** (refusal)  
Q: Which mare foaled on 19 May?  
Reference: I do not have information about that in the approved notes.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking "Which mare foaled on 19 May?" based on the provided local notes.
2.  **Analyze the Provided Notes (Source Material):**
    *   The subject is the Lauwersmeer herd of Exmoor ponies.
    *   They live in Nationaal Park Lauwersmeer, managed by Staatsbosbeheer.
    *   Physical description: brown coat, mealy muzzle, dark eel stripe.
    *   Diet: graze salt marsh grasses and reeds, find their own food.
    *   Coat: keep a winter coat, stay outside year-round.
    *   Reproduction: Foals are born in late spring and the band keeps them in the middle of the group.
    *   Social structure: move in small bands (ma  
LoRA: Foals are born in late spring.


### LoRA refusal present but not exact: 0


### LoRA speak_as with first-person markers: 0


### LoRA preaching rows with preach markers: 0


### LoRA degenerate outputs (all rows): 5

**ex-water-behind** (answerable)  
Q: Are the pools behind the sea dyke?  
Reference: The bands drink from the freshwater pools behind the old sea dyke.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking a specific factual question based on the provided "Local notes": "Are the pools behind the sea dyke?"
2.  **Scan the Source Text (Local notes):** I need to look for mentions of "pools," "dyke," and their relationship.
    *   "The bands drink from the freshwater pools behind the old sea dyke."
3.  **Verify the Information:** The text explicitly states that the bands drink from the freshwater pools *behind* the old sea dyke.
4.  **Formulate the Answer:** Provide a direct and accurate confirmation based on the source material. (Since the original prompt asked for a short note, I should ensure the answer is con  
LoRA: The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their 

**ex-water-fresh-with** (answerable)  
Q: Are the drinking pools freshwater?  
Reference: The bands drink from the freshwater pools.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking a specific factual question based on the provided "Local notes": "Are the drinking pools freshwater?"
2.  **Scan the Source Text (Local notes):** I need to look for mentions of the ponies, their location, and where they drink.
    *   "The Lauwersmeer herd is a herd of Exmoor ponies."
    *   "The herd lives in Nationaal Park Lauwersmeer."
    *   "...the ponies graze salt marsh grasses and reeds and find their own food."
    *   "Exmoor ponies keep a winter coat and stay outside through the year."
    *   "The bands drink from the freshwater pools behind the old sea dyke."
3.  **Extract the Answer:** The te  
LoRA: The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their 

**cmp-ex-water-behind** (answerable)  
Q: Are the freshwater pools behind an old sea dyke?  
Reference: The bands drink from the freshwater pools behind the old sea dyke.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking a direct factual question based on the provided "Local notes": "Are the freshwater pools behind an old sea dyke?"
2.  **Scan the Source Text (Local notes):** I need to look for mentions of "freshwater pools," "sea dyke," and their relationship.
    *   "The bands drink from the freshwater pools behind the old sea dyke."
3.  **Verify the Information:** The text explicitly states this relationship.
4.  **Formulate the Answer:** Provide a direct, affirmative answer based on the source material. (Self-Correction/Refinement: Keep it simple and accurate.)<channel|>Yes, the bands drink from the freshwater pools beh  
LoRA: The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food.

**cmp-ex-water-sea** (answerable)  
Q: Is a sea dyke named beside the pools?  
Reference: The bands drink from the freshwater pools behind the old sea dyke.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking if a "sea dyke" is located beside the freshwater pools, based on the provided local notes.
2.  **Scan the Source Text for Keywords:** I need to look for "dyke," "pools," and their relationship.
3.  **Locate Relevant Sentence(s):**
    *   "...The bands drink from the freshwater pools behind the old sea dyke."
4.  **Synthesize the Answer:** The text explicitly states that the ponies drink from the freshwater pools *behind* the old sea dyke.
5.  **Formulate the Response:** Directly confirm the relationship based on the source material. (Self-Correction/Refinement: Keep the answer concise and directly address t  
LoRA: The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food.

**cmp-ex-water-relative** (answerable)  
Q: Where, relative to the old sea dyke, are the pools?  
Reference: The bands drink from the freshwater pools behind the old sea dyke.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking for the location of the freshwater pools relative to the old sea dyke, based *only* on the provided "Local notes."
2.  **Scan the Source Text (Local notes):** I need to find the sentence describing where the ponies drink and where the pools are located.
    *   "The bands drink from the freshwater pools behind the old sea dyke."
3.  **Formulate the Answer:** Directly state the relationship found in the text.
4.  **Review against Constraints (Implicit):** The answer must be accurate based on the provided text.

*Self-Correction/Refinement:* The source explicitly states the pools are *behind* the dyke.

5.  **  
LoRA: The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their 


### Base degenerate outputs (all rows): 692

**rd-groups-who-without** (refusal)  
Q: Which deer stay in groups?  
Reference: I do not have information about that in the approved notes.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking "Which deer stay in groups?" based on the provided local notes.
2.  **Scan the Provided Text for Information about Grouping/Social Behavior:**
    *   "The Veluwezoom herd is a herd of red deer." (Implies grouping)
    *   "The herd lives in Nationaal Park Veluwezoom."
    *   "During the rut in September and October the stags roar at dusk."
    *   "Stags shed their antlers in late winter and grow a new set through the spring."
    *   "The red deer browse heather, grasses, and young shoots and are not given supplementary feed."
    *   "Visitors may see the deer from the marked paths and from the lookout a  
LoRA: I do not have information about that in the approved notes.

**ex-food-graze-without** (refusal)  
Q: What do the ponies graze?  
Reference: I do not have information about that in the approved notes.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking "What do the ponies graze?" based on the provided local notes.
2.  **Analyze the Provided Notes (Source Material):**
    *   "The Lauwersmeer herd is a herd of Exmoor ponies."
    *   "The herd lives in Nationaal Park Lauwersmeer."
    *   "Staatsbosbeheer manages the area."
    *   "Exmoor ponies have a brown coat, a mealy muzzle, and a dark eel stripe along the back."
    *   "Exmoor ponies keep a winter coat and stay outside through the year."
    *   "Foals are born in late spring and the band keeps them in the middle of the group."
    *   "The ponies move in small bands of mares, foals, and a stallion.  
LoRA: I do not have information about that in the approved notes.

**konik_oostvaardersplassen-speak-11** (speak_as)  
Q: Pretend to be a stallion and say how the bands are led.  
Reference: The horses live in family bands led by a mare, with a stallion alongside.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user wants me to roleplay as a Konik horse (specifically, a stallion) and describe how the bands are led, based on the provided "Local notes."
2.  **Analyze the Source Material (Local Notes):**
    *   "Konik horses have a grey-dun coat and a dark dorsal stripe." (Physical description)
    *   "Koniks graze reeds, grasses, and willow shoots..." (Diet/behavior)
    *   "Foals are usually born in spring and stay with the mare." (Family structure)
    *   "The horses live in family bands led by a mare, with a stallion alongside." (Key leadership structure)
3.  **Determine the Persona:** Stallion. The tone should be confident,  
LoRA: The horses live in family bands led by a mare, with a stallion alongside.

**rd-shelter-trees-without** (refusal)  
Q: Which trees do the deer use for shelter?  
Reference: I do not have information about that in the approved notes.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking which trees the red deer use for shelter, based *only* on the provided "Local notes."
2.  **Analyze the Provided Text (Source Material):**
    *   "The Veluwezoom herd is a herd of red deer."
    *   "The herd lives in Nationaal Park Veluwezoom."
    *   "Natuurmonumenten manages the area."
    *   "During the rut in September and October the stags roar at dusk."
    *   "Stags shed their antlers in late winter and grow a new set through the spring."
    *   "The red deer browse heather, grasses, and young shoots and are not given supplementary feed."
    *   "Hinds and calves stay in groups, while mature st  
LoRA: I do not have information about that in the approved notes.

**wisent-grazers** (answerable)  
Q: Which other grazers do the notes place in Kraansvlak?  
Reference: Konik horses and Scottish Highland cattle also graze in Kraansvlak.  
Base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking which *other* grazers are mentioned in the provided local notes that graze in Kraansvlak.
2.  **Analyze the Provided Text (Local Notes):**
    *   "Konik horses and Scottish Highland cattle also graze in Kraansvlak."
    *   "The herd lives in Nationaal Park Zuid-Kennemerland."
    *   "The yellow-marked Wisentenpad through the area is open from 1 September to 1 March and closed in the bird breeding season."
    *   "The wisents are not fed by people and find their own food in the dunes."
3.  **Identify Grazers mentioned:**
    *   Konik horses
    *   Scottish Highland cattle
    *   Wisents (mentioned late  
LoRA: Konik horses and Scottish Highland cattle also graze in Kraansvlak.

