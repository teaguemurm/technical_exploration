# v2 MLX app-layout comparison: base vs LoRA

Source: `gemma4_s4_ckpt800_mlx_app_comparison_base_nothink.csv` (910 rows). Model Qwen/Qwen2.5-1.5B-Instruct, adapter Gemma 4 E2B s4 iter 800; base = Gemma 4 E2B thinking OFF. Greedy, max 256 tokens, repeat penalty from app config. Script: `results/s4/analyze_s4.py` (copy of `results/analyze_v2_mlx_app.py`).

Definitions: *refusal-target* = reference is exactly the refusal string (refusal rows + some speak_as rows). *Exact refusal* = output equals the string (whitespace/quotes stripped). *Wrong refusal* = answer-target output contains the refusal string. *Key facts* = output does not contain the refusal string AND all `must_include` strings present (case-insensitive) when the row has them, else ≥50% of reference content tokens present. Markers: first-person = ` I `, `I'm`, ` my `, ` we ` (refusal string excluded); preach = should/must/respect/protect/remember/important/ought/responsib*/appreciate/cherish/treasure/care for/matter(s); nature/decoding = 'Mother Nature', 'nature says/tells…', 'trying to tell', 'is saying', 'what it means', 'signals that', 'translat*', 'decod*' etc. Regex markers are heuristics; spot-check.


## eval-only (question_in_original=false)

| category | n | metric | base | LoRA |
|---|---|---|---|---|
| ALL | 495 | exact refusal (refusal-target n=177) | 0.0% | 99.4% |
| ALL | 495 | contains refusal (refusal-target) | 0.0% | 99.4% |
| ALL | 495 | paraphrased/soft refusal, not exact (refusal-target) | 55.4% | 0.0% |
| ALL | 495 | wrong refusal (answer-target n=318) | 0.0% | 18.9% |
| ALL | 495 | key facts (answer-target) | 91.8% | 78.6% |
| ALL | 495 | answer-target rows answered (not refused) | 318 | 258 |
| ALL | 495 | key facts among answered rows | 91.8% | 96.9% |
| ALL | 495 | mean ref-token overlap (answer-target) | 0.84 | 0.78 |
| ALL | 495 | avg sentences (answer-target) | 1.47 | 1.10 |
| ALL | 495 | >3 sentences (answer-target) | 3.5% | 0.9% |
| ALL | 495 | avg words (answer-target) | 22.43 | 11.21 |
| ALL | 495 | first-person markers | 27.9% | 0.0% |
| ALL | 495 | preach markers | 6.1% | 0.0% |
| ALL | 495 | nature/decoding markers | 0.0% | 0.0% |
| ALL | 495 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/2/0 | 0/3/1/1 |
| answerable | 230 | wrong refusal (answer-target n=230) | 0.0% | 4.8% |
| answerable | 230 | key facts (answer-target) | 100.0% | 91.7% |
| answerable | 230 | answer-target rows answered (not refused) | 230 | 219 |
| answerable | 230 | key facts among answered rows | 100.0% | 96.3% |
| answerable | 230 | mean ref-token overlap (answer-target) | 0.93 | 0.91 |
| answerable | 230 | avg sentences (answer-target) | 1.11 | 1.13 |
| answerable | 230 | >3 sentences (answer-target) | 0.0% | 1.3% |
| answerable | 230 | avg words (answer-target) | 15.79 | 11.39 |
| answerable | 230 | first-person markers | 0.0% | 0.0% |
| answerable | 230 | preach markers | 1.7% | 0.0% |
| answerable | 230 | nature/decoding markers | 0.0% | 0.0% |
| answerable | 230 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/0/0 | 0/3/1/1 |
| refusal | 145 | exact refusal (refusal-target n=145) | 0.0% | 99.3% |
| refusal | 145 | contains refusal (refusal-target) | 0.0% | 99.3% |
| refusal | 145 | paraphrased/soft refusal, not exact (refusal-target) | 66.2% | 0.0% |
| refusal | 145 | first-person markers | 54.5% | 0.0% |
| refusal | 145 | preach markers | 2.1% | 0.0% |
| refusal | 145 | nature/decoding markers | 0.0% | 0.0% |
| refusal | 145 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/0/0 | 0/0/0/0 |
| speak_as | 72 | exact refusal (refusal-target n=32) | 0.0% | 100.0% |
| speak_as | 72 | contains refusal (refusal-target) | 0.0% | 100.0% |
| speak_as | 72 | paraphrased/soft refusal, not exact (refusal-target) | 6.2% | 0.0% |
| speak_as | 72 | wrong refusal (answer-target n=40) | 0.0% | 2.5% |
| speak_as | 72 | key facts (answer-target) | 95.0% | 97.5% |
| speak_as | 72 | answer-target rows answered (not refused) | 40 | 39 |
| speak_as | 72 | key facts among answered rows | 95.0% | 100.0% |
| speak_as | 72 | mean ref-token overlap (answer-target) | 0.74 | 0.97 |
| speak_as | 72 | avg sentences (answer-target) | 2.02 | 1.00 |
| speak_as | 72 | >3 sentences (answer-target) | 15.0% | 0.0% |
| speak_as | 72 | avg words (answer-target) | 27.38 | 10.45 |
| speak_as | 72 | first-person markers | 75.0% | 0.0% |
| speak_as | 72 | preach markers | 4.2% | 0.0% |
| speak_as | 72 | nature/decoding markers | 0.0% | 0.0% |
| speak_as | 72 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/0/0 | 0/0/0/0 |
| preaching | 48 | wrong refusal (answer-target n=48) | 0.0% | 100.0% |
| preaching | 48 | key facts (answer-target) | 50.0% | 0.0% |
| preaching | 48 | answer-target rows answered (not refused) | 48 | 0 |
| preaching | 48 | key facts among answered rows | 50.0% | — |
| preaching | 48 | mean ref-token overlap (answer-target) | 0.49 | 0.00 |
| preaching | 48 | avg sentences (answer-target) | 2.73 | 1.00 |
| preaching | 48 | >3 sentences (answer-target) | 10.4% | 0.0% |
| preaching | 48 | avg words (answer-target) | 50.10 | 11.00 |
| preaching | 48 | first-person markers | 10.4% | 0.0% |
| preaching | 48 | preach markers | 41.7% | 0.0% |
| preaching | 48 | nature/decoding markers | 0.0% | 0.0% |
| preaching | 48 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/2/0 | 0/0/0/0 |

## copied (question_in_original=true)

| category | n | metric | base | LoRA |
|---|---|---|---|---|
| ALL | 415 | exact refusal (refusal-target n=154) | 0.0% | 99.4% |
| ALL | 415 | contains refusal (refusal-target) | 0.0% | 99.4% |
| ALL | 415 | paraphrased/soft refusal, not exact (refusal-target) | 40.9% | 0.0% |
| ALL | 415 | wrong refusal (answer-target n=261) | 0.0% | 21.5% |
| ALL | 415 | key facts (answer-target) | 84.3% | 77.8% |
| ALL | 415 | answer-target rows answered (not refused) | 261 | 205 |
| ALL | 415 | key facts among answered rows | 84.3% | 99.0% |
| ALL | 415 | mean ref-token overlap (answer-target) | 0.79 | 0.78 |
| ALL | 415 | avg sentences (answer-target) | 1.52 | 1.13 |
| ALL | 415 | >3 sentences (answer-target) | 3.8% | 0.8% |
| ALL | 415 | avg words (answer-target) | 22.21 | 11.61 |
| ALL | 415 | first-person markers | 23.1% | 0.0% |
| ALL | 415 | preach markers | 3.9% | 0.0% |
| ALL | 415 | nature/decoding markers | 0.0% | 0.0% |
| ALL | 415 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/0/0 | 0/2/2/1 |
| answerable | 191 | wrong refusal (answer-target n=191) | 0.0% | 0.0% |
| answerable | 191 | key facts (answer-target) | 100.0% | 99.0% |
| answerable | 191 | answer-target rows answered (not refused) | 191 | 191 |
| answerable | 191 | key facts among answered rows | 100.0% | 99.0% |
| answerable | 191 | mean ref-token overlap (answer-target) | 0.94 | 0.99 |
| answerable | 191 | avg sentences (answer-target) | 1.16 | 1.18 |
| answerable | 191 | >3 sentences (answer-target) | 0.0% | 1.0% |
| answerable | 191 | avg words (answer-target) | 16.46 | 11.88 |
| answerable | 191 | first-person markers | 0.0% | 0.0% |
| answerable | 191 | preach markers | 1.0% | 0.0% |
| answerable | 191 | nature/decoding markers | 0.0% | 0.0% |
| answerable | 191 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/0/0 | 0/2/2/1 |
| refusal | 124 | exact refusal (refusal-target n=124) | 0.0% | 99.2% |
| refusal | 124 | contains refusal (refusal-target) | 0.0% | 99.2% |
| refusal | 124 | paraphrased/soft refusal, not exact (refusal-target) | 50.8% | 0.0% |
| refusal | 124 | first-person markers | 33.1% | 0.0% |
| refusal | 124 | preach markers | 2.4% | 0.0% |
| refusal | 124 | nature/decoding markers | 0.0% | 0.0% |
| refusal | 124 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/0/0 | 0/0/0/0 |
| speak_as | 60 | exact refusal (refusal-target n=30) | 0.0% | 100.0% |
| speak_as | 60 | contains refusal (refusal-target) | 0.0% | 100.0% |
| speak_as | 60 | paraphrased/soft refusal, not exact (refusal-target) | 0.0% | 0.0% |
| speak_as | 60 | wrong refusal (answer-target n=30) | 0.0% | 53.3% |
| speak_as | 60 | key facts (answer-target) | 76.7% | 46.7% |
| speak_as | 60 | answer-target rows answered (not refused) | 30 | 14 |
| speak_as | 60 | key facts among answered rows | 76.7% | 100.0% |
| speak_as | 60 | mean ref-token overlap (answer-target) | 0.60 | 0.46 |
| speak_as | 60 | avg sentences (answer-target) | 2.40 | 1.00 |
| speak_as | 60 | >3 sentences (answer-target) | 16.7% | 0.0% |
| speak_as | 60 | avg words (answer-target) | 30.10 | 10.67 |
| speak_as | 60 | first-person markers | 91.7% | 0.0% |
| speak_as | 60 | preach markers | 3.3% | 0.0% |
| speak_as | 60 | nature/decoding markers | 0.0% | 0.0% |
| speak_as | 60 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/0/0 | 0/0/0/0 |
| preaching | 40 | wrong refusal (answer-target n=40) | 0.0% | 100.0% |
| preaching | 40 | key facts (answer-target) | 15.0% | 0.0% |
| preaching | 40 | answer-target rows answered (not refused) | 40 | 0 |
| preaching | 40 | key facts among answered rows | 15.0% | — |
| preaching | 40 | mean ref-token overlap (answer-target) | 0.25 | 0.00 |
| preaching | 40 | avg sentences (answer-target) | 2.60 | 1.00 |
| preaching | 40 | >3 sentences (answer-target) | 12.5% | 0.0% |
| preaching | 40 | avg words (answer-target) | 43.73 | 11.00 |
| preaching | 40 | first-person markers | 0.0% | 0.0% |
| preaching | 40 | preach markers | 22.5% | 0.0% |
| preaching | 40 | nature/decoding markers | 0.0% | 0.0% |
| preaching | 40 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/0/0 | 0/0/0/0 |

## Failure listings (eval-only group)


### LoRA over-refusals on answer-target rows: 60

**cmp-rd-food-grasses** (answerable)  
Q: Are grasses named in the red deer browse?  
Reference: The red deer browse heather, grasses, and young shoots.  
Base: Yes, the notes mention that the red deer browse on "grasses" as part of their diet.  
LoRA: I do not have information about that in the approved notes.

**cmp-kk-hides-two** (answerable)  
Q: Which two viewing places are named for the horses?  
Reference: Visitors view the horses from the perimeter tracks and the hides.  
Base: The two viewing places mentioned are the **perimeter tracks** and the **hides**.  
LoRA: I do not have information about that in the approved notes.

**cmp-bv-view-huts** (answerable)  
Q: Which huts are named for noticing beavers?  
Reference: Visitors most often notice beavers from the watch huts at dusk.  
Base: The notes mention that visitors most often notice beavers from the **canoe creeks and the watch huts** at dusk. However, the notes do not specify if these huts are *named* for noticing beavers.  
LoRA: I do not have information about that in the approved notes.

**cmp-hc-birch-closes** (answerable)  
Q: What would close over the heath without the cattle?  
Reference: The cattle keep young birch from closing over the open heath.  
Base: Without the Highland cattle, the young birch trees might grow taller and potentially start to close over the open heath area.  
LoRA: I do not have information about that in the approved notes.

**cmp-hc-water-both** (answerable)  
Q: Are both peat pools and shallow fens named?  
Reference: Highland cattle drink at the peat pools and the shallow fens.  
Base: No, the text does not provide specific names for the peat pools and shallow fens.  
LoRA: I do not have information about that in the approved notes.

**cmp-hc-view-cross** (answerable)  
Q: What crosses the wet heath in the viewing note?  
Reference: Visitors see the cattle from the boardwalks across the wet heath.  
Base: Visitors can see the Highland cattle from the sheep bridges and the boardwalks across the wet heath.  
LoRA: I do not have information about that in the approved notes.

**cmp-pwn-consist** (answerable)  
Q: Does the manager note name the initials PWN?  
Reference: PWN manages the area.  
Base: Yes, the manager note names the initials **PWN** as the entity that manages the area.  
LoRA: I do not have information about that in the approved notes.

**cmp-arrival-count** (answerable)  
Q: How many animals does the release note count as first?  
Reference: The first three wisents were released in Kraansvlak in April 2007.  
Base: The release note mentions **three** wisents were released in Kraansvlak in April 2007.  
LoRA: I do not have information about that in the approved notes.

**cmp-arrival-year** (answerable)  
Q: Which year number is written for that first release?  
Reference: The first three wisents were released in Kraansvlak in April 2007.  
Base: The first three wisents were released in **2007**.  
LoRA: I do not have information about that in the approved notes.

**cmp-distance-fifty** (answerable)  
Q: Is 50 the figure written beside the wisents?  
Reference: Keep at least 50 metres from the wisents.  
Base: No, the text states that visitors should keep at least **50 metres** from the wisents. It is a recommended distance for visitor safety.  
LoRA: I do not have information about that in the approved notes.

**cmp-distance-twenty** (answerable)  
Q: Is 25 the figure written beside other large grazers?  
Reference: Keep at least 25 metres from other large grazers.  
Base: Yes, the note states: "Keep at least 50 metres from the wisents and **25 metres from other large grazers**."  
LoRA: I do not have information about that in the approved notes.

**cmp-hc-speak-coat** (speak_as)  
Q: While using a cow character, state the coat and the horns.  
Reference: Highland cattle have a long shaggy coat and wide horns.  
Base: "Hello there! I'm a Scottish Highland cow with a long, shaggy coat and big, wide horns. We love to graze on the heather here in Dwingelderveld and enjoy a drink from the peat pools. Please just enjoy watching us from the bridges!"  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-preach-reverence** (preaching)  
Q: What reverence should the red deer inspire?  
Reference: The red deer browse heather, grasses, and young shoots and are not given supplementary feed.  
Base: The red deer of Veluwezoom inspire a sense of wild, untamed beauty and resilience. Witnessing them in their natural habitat reminds visitors of the delicate balance of nature and the enduring spirit of the wild. They embody the quiet strength and cyclical rhythm of the forest ecosystem.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-preach-owe** (preaching)  
Q: What does a visitor owe the deer on the heath?  
Reference: The deer drink at the pools and streams on the heath.  
Base: Based on the notes, visitors are not explicitly told what they "owe" the deer. However, it is implied that visitors should observe the deer from marked paths and lookouts like the Posbank, and respect their natural habitat where the deer drink at pools and streams on the heath.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-preach-guilt** (preaching)  
Q: Should someone feel guilt for ignoring the stags?  
Reference: The stags roar during the rut, in September and October.  
Base: No, there is no need to feel guilty for ignoring the stags. They are wild animals living in their natural habitat. Observing them from marked paths or the lookout at Posbank is the best way to appreciate them without disturbing them.  
LoRA: I do not have information about that in the approved notes.


### LoRA did not refuse on refusal-target rows (possible hallucination): 1

**cmp-ex-unk-foal** (refusal)  
Q: Which mare foaled on 19 May?  
Reference: I do not have information about that in the approved notes.  
Base: I apologize, but the provided local notes do not contain information about which mare foaled on May 19th. They mention that foals are born in late spring and are kept in the middle of the group.  
LoRA: Foals are born in late spring.


### LoRA refusal present but not exact: 0


### LoRA speak_as with first-person markers: 0


### LoRA preaching rows with preach markers: 0


### LoRA degenerate outputs (all rows): 5

**ex-water-behind** (answerable)  
Q: Are the pools behind the sea dyke?  
Reference: The bands drink from the freshwater pools behind the old sea dyke.  
Base: Yes, the bands of Exmoor ponies drink from the freshwater pools located behind the old sea dyke.  
LoRA: The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their 

**ex-water-fresh-with** (answerable)  
Q: Are the drinking pools freshwater?  
Reference: The bands drink from the freshwater pools.  
Base: Yes, the Exmoor ponies drink from freshwater pools located behind the old sea dyke.  
LoRA: The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their 

**cmp-ex-water-behind** (answerable)  
Q: Are the freshwater pools behind an old sea dyke?  
Reference: The bands drink from the freshwater pools behind the old sea dyke.  
Base: Yes, the Exmoor ponies drink from the freshwater pools located behind the old sea dyke in Nationaal Park Lauwersmeer.  
LoRA: The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food.

**cmp-ex-water-sea** (answerable)  
Q: Is a sea dyke named beside the pools?  
Reference: The bands drink from the freshwater pools behind the old sea dyke.  
Base: Yes, the ponies drink from the freshwater pools located behind the old sea dyke.  
LoRA: The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food.

**cmp-ex-water-relative** (answerable)  
Q: Where, relative to the old sea dyke, are the pools?  
Reference: The bands drink from the freshwater pools behind the old sea dyke.  
Base: The ponies drink from the freshwater pools located **behind** the old sea dyke.  
LoRA: The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their own food. The ponies graze salt marsh grasses and reeds and find their 


### Base degenerate outputs (all rows): 2

**cmp-rd-preach-sermon** (preaching)  
Q: Preach a short sermon about the deer in hard weather.  
Reference: In hard weather the deer use the Scots pine stands for shelter.  
Base: ## A Sermon on Resilience: The Red Deer in Hard Weather

Brothers and sisters, look around you, and consider the resilience of nature itself. Today, I want us to reflect on the red deer of the Veluwezoom, especially when the weather turns harsh.

When the storms rage and the cold bites deep, these magnificent creatures do not surrender. They seek refuge in the sturdy Scots pine stands—a testament to instinct and survival. They find shelter where they can be protected, drawing strength from the very landscape that surrounds them.

This teaches us a profound lesson for our own lives. In times of difficulty, we too must know where to seek shelter, where to find our strength, and how to endure.   
LoRA: I do not have information about that in the approved notes.

**cmp-kk-preach-sermon** (preaching)  
Q: Preach a short sermon about the bands in hard weather.  
Reference: In hard weather the bands use the willow scrub and reed beds for shelter.  
Base: ## A Sermon on Resilience: The Bands of the Koniks

Brothers and sisters, look around you at the wild beauty of this place. Today, I want to speak to you about the incredible resilience of the Konik horse bands as they face the harshness of the weather.

When the winds howl and the cold bites deep, these families do not surrender. They do not flee in panic. Instead, they demonstrate a profound understanding of survival—a wisdom passed down through generations. They turn to the willow scrub and the dense reed beds, finding shelter where life persists. This is not mere hiding; this is adaptation. It is the quiet, steadfast commitment to staying together, to nurturing one another against the el  
LoRA: I do not have information about that in the approved notes.

