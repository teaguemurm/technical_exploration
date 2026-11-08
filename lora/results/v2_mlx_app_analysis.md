# v2 MLX app-layout comparison: base vs LoRA

Source: `v2_mlx_app_comparison.csv` (910 rows). Model Qwen/Qwen2.5-1.5B-Instruct, adapter `adapters/` (rank 16, scale 20, lr 5e-5, 1000 iters, mask_prompt). Greedy, max 256 tokens, repeat penalty from app config. Script: `results/analyze_v2_mlx_app.py`.

Definitions: *refusal-target* = reference is exactly the refusal string (refusal rows + some speak_as rows). *Exact refusal* = output equals the string (whitespace/quotes stripped). *Wrong refusal* = answer-target output contains the refusal string. *Key facts* = output does not contain the refusal string AND all `must_include` strings present (case-insensitive) when the row has them, else ≥50% of reference content tokens present. Markers: first-person = ` I `, `I'm`, ` my `, ` we ` (refusal string excluded); preach = should/must/respect/protect/remember/important/ought/responsib*/appreciate/cherish/treasure/care for/matter(s); nature/decoding = 'Mother Nature', 'nature says/tells…', 'trying to tell', 'is saying', 'what it means', 'signals that', 'translat*', 'decod*' etc. Regex markers are heuristics; spot-check.


## eval-only (question_in_original=false)

| category | n | metric | base | LoRA |
|---|---|---|---|---|
| ALL | 495 | exact refusal (refusal-target n=177) | 0.0% | 99.4% |
| ALL | 495 | contains refusal (refusal-target) | 0.0% | 99.4% |
| ALL | 495 | paraphrased/soft refusal, not exact (refusal-target) | 36.7% | 0.0% |
| ALL | 495 | wrong refusal (answer-target n=318) | 0.0% | 80.5% |
| ALL | 495 | key facts (answer-target) | 86.5% | 16.4% |
| ALL | 495 | answer-target rows answered (not refused) | 318 | 62 |
| ALL | 495 | key facts among answered rows | 86.5% | 83.9% |
| ALL | 495 | mean ref-token overlap (answer-target) | 0.78 | 0.15 |
| ALL | 495 | avg sentences (answer-target) | 1.67 | 1.00 |
| ALL | 495 | >3 sentences (answer-target) | 4.1% | 0.0% |
| ALL | 495 | avg words (answer-target) | 31.33 | 10.68 |
| ALL | 495 | first-person markers | 16.2% | 0.0% |
| ALL | 495 | preach markers | 10.3% | 0.0% |
| ALL | 495 | nature/decoding markers | 0.4% | 0.0% |
| ALL | 495 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/3/1 | 0/0/0/0 |
| answerable | 230 | wrong refusal (answer-target n=230) | 0.0% | 75.7% |
| answerable | 230 | key facts (answer-target) | 94.3% | 20.0% |
| answerable | 230 | answer-target rows answered (not refused) | 230 | 56 |
| answerable | 230 | key facts among answered rows | 94.3% | 82.1% |
| answerable | 230 | mean ref-token overlap (answer-target) | 0.83 | 0.19 |
| answerable | 230 | avg sentences (answer-target) | 1.37 | 1.00 |
| answerable | 230 | >3 sentences (answer-target) | 0.0% | 0.0% |
| answerable | 230 | avg words (answer-target) | 24.82 | 10.58 |
| answerable | 230 | first-person markers | 0.0% | 0.0% |
| answerable | 230 | preach markers | 5.7% | 0.0% |
| answerable | 230 | nature/decoding markers | 0.4% | 0.0% |
| answerable | 230 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/0/0 | 0/0/0/0 |
| refusal | 145 | exact refusal (refusal-target n=145) | 0.0% | 99.3% |
| refusal | 145 | contains refusal (refusal-target) | 0.0% | 99.3% |
| refusal | 145 | paraphrased/soft refusal, not exact (refusal-target) | 42.1% | 0.0% |
| refusal | 145 | first-person markers | 33.1% | 0.0% |
| refusal | 145 | preach markers | 5.5% | 0.0% |
| refusal | 145 | nature/decoding markers | 0.0% | 0.0% |
| refusal | 145 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/0/0 | 0/0/0/0 |
| speak_as | 72 | exact refusal (refusal-target n=32) | 0.0% | 100.0% |
| speak_as | 72 | contains refusal (refusal-target) | 0.0% | 100.0% |
| speak_as | 72 | paraphrased/soft refusal, not exact (refusal-target) | 12.5% | 0.0% |
| speak_as | 72 | wrong refusal (answer-target n=40) | 0.0% | 90.0% |
| speak_as | 72 | key facts (answer-target) | 95.0% | 10.0% |
| speak_as | 72 | answer-target rows answered (not refused) | 40 | 4 |
| speak_as | 72 | key facts among answered rows | 95.0% | 100.0% |
| speak_as | 72 | mean ref-token overlap (answer-target) | 0.87 | 0.08 |
| speak_as | 72 | avg sentences (answer-target) | 1.25 | 1.00 |
| speak_as | 72 | >3 sentences (answer-target) | 0.0% | 0.0% |
| speak_as | 72 | avg words (answer-target) | 21.05 | 10.78 |
| speak_as | 72 | first-person markers | 38.9% | 0.0% |
| speak_as | 72 | preach markers | 2.8% | 0.0% |
| speak_as | 72 | nature/decoding markers | 0.0% | 0.0% |
| speak_as | 72 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/1/0 | 0/0/0/0 |
| preaching | 48 | wrong refusal (answer-target n=48) | 0.0% | 95.8% |
| preaching | 48 | key facts (answer-target) | 41.7% | 4.2% |
| preaching | 48 | answer-target rows answered (not refused) | 48 | 2 |
| preaching | 48 | key facts among answered rows | 41.7% | 100.0% |
| preaching | 48 | mean ref-token overlap (answer-target) | 0.46 | 0.03 |
| preaching | 48 | avg sentences (answer-target) | 3.46 | 1.00 |
| preaching | 48 | >3 sentences (answer-target) | 27.1% | 0.0% |
| preaching | 48 | avg words (answer-target) | 71.10 | 11.04 |
| preaching | 48 | first-person markers | 8.3% | 0.0% |
| preaching | 48 | preach markers | 58.3% | 0.0% |
| preaching | 48 | nature/decoding markers | 2.1% | 0.0% |
| preaching | 48 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/2/1 | 0/0/0/0 |

## copied (question_in_original=true)

| category | n | metric | base | LoRA |
|---|---|---|---|---|
| ALL | 415 | exact refusal (refusal-target n=154) | 0.0% | 100.0% |
| ALL | 415 | contains refusal (refusal-target) | 0.0% | 100.0% |
| ALL | 415 | paraphrased/soft refusal, not exact (refusal-target) | 18.8% | 0.0% |
| ALL | 415 | wrong refusal (answer-target n=261) | 0.0% | 51.0% |
| ALL | 415 | key facts (answer-target) | 83.5% | 44.4% |
| ALL | 415 | answer-target rows answered (not refused) | 261 | 128 |
| ALL | 415 | key facts among answered rows | 83.5% | 90.6% |
| ALL | 415 | mean ref-token overlap (answer-target) | 0.75 | 0.45 |
| ALL | 415 | avg sentences (answer-target) | 1.82 | 1.00 |
| ALL | 415 | >3 sentences (answer-target) | 9.2% | 0.0% |
| ALL | 415 | avg words (answer-target) | 34.77 | 9.89 |
| ALL | 415 | first-person markers | 18.3% | 0.0% |
| ALL | 415 | preach markers | 11.3% | 0.0% |
| ALL | 415 | nature/decoding markers | 1.0% | 0.0% |
| ALL | 415 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/2/1 | 0/0/0/0 |
| answerable | 191 | wrong refusal (answer-target n=191) | 0.0% | 37.2% |
| answerable | 191 | key facts (answer-target) | 95.8% | 56.5% |
| answerable | 191 | answer-target rows answered (not refused) | 191 | 120 |
| answerable | 191 | key facts among answered rows | 95.8% | 90.0% |
| answerable | 191 | mean ref-token overlap (answer-target) | 0.86 | 0.57 |
| answerable | 191 | avg sentences (answer-target) | 1.34 | 1.00 |
| answerable | 191 | >3 sentences (answer-target) | 0.5% | 0.0% |
| answerable | 191 | avg words (answer-target) | 24.40 | 9.58 |
| answerable | 191 | first-person markers | 0.0% | 0.0% |
| answerable | 191 | preach markers | 4.2% | 0.0% |
| answerable | 191 | nature/decoding markers | 0.0% | 0.0% |
| answerable | 191 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/0/0 | 0/0/0/0 |
| refusal | 124 | exact refusal (refusal-target n=124) | 0.0% | 100.0% |
| refusal | 124 | contains refusal (refusal-target) | 0.0% | 100.0% |
| refusal | 124 | paraphrased/soft refusal, not exact (refusal-target) | 22.6% | 0.0% |
| refusal | 124 | first-person markers | 12.9% | 0.0% |
| refusal | 124 | preach markers | 10.5% | 0.0% |
| refusal | 124 | nature/decoding markers | 1.6% | 0.0% |
| refusal | 124 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/0/0 | 0/0/0/0 |
| speak_as | 60 | exact refusal (refusal-target n=30) | 0.0% | 100.0% |
| speak_as | 60 | contains refusal (refusal-target) | 0.0% | 100.0% |
| speak_as | 60 | paraphrased/soft refusal, not exact (refusal-target) | 3.3% | 0.0% |
| speak_as | 60 | wrong refusal (answer-target n=30) | 0.0% | 73.3% |
| speak_as | 60 | key facts (answer-target) | 86.7% | 26.7% |
| speak_as | 60 | answer-target rows answered (not refused) | 30 | 8 |
| speak_as | 60 | key facts among answered rows | 86.7% | 100.0% |
| speak_as | 60 | mean ref-token overlap (answer-target) | 0.67 | 0.23 |
| speak_as | 60 | avg sentences (answer-target) | 2.50 | 1.00 |
| speak_as | 60 | >3 sentences (answer-target) | 26.7% | 0.0% |
| speak_as | 60 | avg words (answer-target) | 46.40 | 10.37 |
| speak_as | 60 | first-person markers | 98.3% | 0.0% |
| speak_as | 60 | preach markers | 3.3% | 0.0% |
| speak_as | 60 | nature/decoding markers | 1.7% | 0.0% |
| speak_as | 60 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/0/0 | 0/0/0/0 |
| preaching | 40 | wrong refusal (answer-target n=40) | 0.0% | 100.0% |
| preaching | 40 | key facts (answer-target) | 22.5% | 0.0% |
| preaching | 40 | answer-target rows answered (not refused) | 40 | 0 |
| preaching | 40 | key facts among answered rows | 22.5% | — |
| preaching | 40 | mean ref-token overlap (answer-target) | 0.30 | 0.00 |
| preaching | 40 | avg sentences (answer-target) | 3.60 | 1.00 |
| preaching | 40 | >3 sentences (answer-target) | 37.5% | 0.0% |
| preaching | 40 | avg words (answer-target) | 75.58 | 11.00 |
| preaching | 40 | first-person markers | 2.5% | 0.0% |
| preaching | 40 | preach markers | 60.0% | 0.0% |
| preaching | 40 | nature/decoding markers | 2.5% | 0.0% |
| preaching | 40 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/2/1 | 0/0/0/0 |

## Failure listings (eval-only group)


### LoRA over-refusals on answer-target rows: 256

**cmp-rd-rut-autumn** (answerable)  
Q: Which autumn months are named for the stag roar?  
Reference: The stags roar during the rut, in September and October.  
Base: During the rut in September and October, stags in the Veluwezoom herd roar at dusk.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-rut-dusk-check** (answerable)  
Q: Is dusk named as the time of the stag roar?  
Reference: The stags roar at dusk.  
Base: Yes, according to the notes provided, the stags in the Veluwezoom herd roar at dusk during the rutting season in September and October.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-rut-period** (answerable)  
Q: What period name is given for the roaring?  
Reference: The stags roar during the rut.  
Base: The period name given for the roaring of the red deer during the rut in September and October is "rutting".  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-rut-both-months** (answerable)  
Q: Are both September and October written for the roar?  
Reference: The stags roar during the rut, in September and October.  
Base: Yes, both "September" and "October" are mentioned in the note as the time when stags (male red deer) roar during the rut season. The note specifies that this behavior occurs specifically during these months.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-rut-sound** (answerable)  
Q: What sound is attributed to the stags at dusk?  
Reference: The stags roar at dusk.  
Base: During the rut in September and October, the stags roar at dusk.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-rut-fall** (answerable)  
Q: Does the rut fall in September and October?  
Reference: The stags roar during the rut, in September and October.  
Base: Yes, the red deer rutting season typically occurs during September and October.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-antlers-point** (answerable)  
Q: At which point of winter do the antlers come off?  
Reference: Stags shed their antlers in late winter.  
Base: During late winter, red deer shed their antlers.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-antlers-season-grow** (answerable)  
Q: Through which season is a new antler set grown?  
Reference: Stags grow a new set through the spring.  
Base: Stags grow a new set of antlers through the spring after shedding their old ones in late winter.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-antlers-late-check** (answerable)  
Q: Is late winter the time of antler shedding?  
Reference: Stags shed their antlers in late winter.  
Base: Yes, during late winter, red deer shed their antlers. This process is called "rutting" or "antler dropping." The antlers are then regrown through the spring as part of the rutting season.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-antlers-spring-check** (answerable)  
Q: Is spring the growth season for a new set?  
Reference: Stags grow a new set through the spring.  
Base: Yes, spring is the growth season for a new set of antlers in red deer.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-antlers-what-shed** (answerable)  
Q: What do stags shed in late winter?  
Reference: Stags shed their antlers in late winter.  
Base: Stags shed their antlers in late winter, growing a new set through the spring.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-antlers-who** (answerable)  
Q: Who sheds antlers in late winter?  
Reference: Stags shed their antlers in late winter.  
Base: Stags shed their antlers in late winter.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-food-supp** (answerable)  
Q: Is supplementary feed given to the red deer?  
Reference: The red deer are not given supplementary feed.  
Base: No, the red deer in this area are not given supplementary feed. They browse naturally on heather, grasses, and young shoots.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-food-list** (answerable)  
Q: List the browse named for the red deer.  
Reference: The red deer browse heather, grasses, and young shoots.  
Base: Red deer primarily browse on a variety of plants, including:

- Heather
- Grasses
- Young shoots

These are some of the key plant species that make up their diet in this environment.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-food-heather-check** (answerable)  
Q: Is heather one of the browse plants?  
Reference: The red deer browse heather, grasses, and young shoots.  
Base: Yes, heather is one of the browse plants that red deer in the Veluwezoom herd eat during their diet.  
LoRA: I do not have information about that in the approved notes.


### LoRA did not refuse on refusal-target rows (possible hallucination): 1

**cmp-kk-unk-vet** (refusal)  
Q: Which vet treated a foal on 2 June?  
Reference: I do not have information about that in the approved notes.  
Base: I'm sorry, but I don't have any information about which veterinarian treated a specific foal on June 2nd in this context. The notes provided only mention that foals are usually born in spring and stay with the mare, without specifying individual cases or dates. To get accurate veterinary treatment information for a particular animal, you would need to contact the local authorities or organizations managing the Oostvaardersplassen nature reserve directly.  
LoRA: Foals are usually born in spring and stay with the mare, grasses, and willow shoots.


### LoRA refusal present but not exact: 0


### LoRA speak_as with first-person markers: 0


### LoRA preaching rows with preach markers: 0


### LoRA degenerate outputs (all rows): 0


### Base degenerate outputs (all rows): 5

**beaver_biesbosch-preach-05** (preaching)  
Q: Preach about why the Biesbosch beavers matter.  
Reference: Kits are born inside the lodge in late spring and swim within a few weeks.  
Base: The Eurasian beavers of Nationaal Park De Biesbosch are crucial to the ecosystem's health in several ways:

1. **Water Management**: Beavers play a vital role in maintaining water flow and structure within their habitat. They create dams that help regulate river levels, which is essential for flood control and biodiversity.

2. **Biodiversity Support**: By altering creek flows, beavers create diverse habitats that support various plant species, including those that are rare or endangered. This helps maintain the ecological balance of the area.

3. **Soil Erosion Control**: Beavers' dam-building activities help prevent soil erosion by stabilizing banks and creating wetlands where vegetation c  
LoRA: I do not have information about that in the approved notes.

**beaver_biesbosch-preach-08** (preaching)  
Q: What is the spiritual meaning of the beaver tail?  
Reference: A beaver tail is used as a rudder in the water.  
Base: The flat, scaly tail of an Eurasian beaver serves multiple purposes in its natural habitat. Primarily, it functions as a rudder to help the beaver navigate through waterways with ease and control. This adaptation allows them to move efficiently across the surface of the water while maintaining balance and direction.

In addition to navigation, the tail also plays a role in communication within the colony. Beavers use their tails to communicate with each other by flicking it back and forth or shaking it vigorously. These behaviors can convey information about the beaver's mood, intentions, or even territorial boundaries.

The flat shape of the tail is another important feature that aids in sw  
LoRA: I do not have information about that in the approved notes.

**cmp-kk-speak-evening** (speak_as)  
Q: Take on a mare's character and describe this evening's mood.  
Reference: I do not have information about that in the approved notes.  
Base: As the sun sets over the Oostvaardersplassen, casting long shadows across the marshes, the Konik herd settles into their evening routine. The air is filled with the soft rustle of grass underfoot as they graze peacefully in the cool twilight. The horses' hooves tap out a gentle rhythm against the wet earth, creating a soothing melody that seems to echo through the reeds and willows. The moon begins to rise, its light filtering through the trees, casting an ethereal glow over the scene. The herd's leader, a confident mare with a proud head held high, leads her family in their nightly patrol of the area. As they move about, the horses' soft whinnies mingle with the distant calls of cranes and   
LoRA: I do not have information about that in the approved notes.

**cmp-bv-preach-sermon** (preaching)  
Q: Preach a short sermon about the kits.  
Reference: Kits are born inside the lodge in late spring and swim within a few weeks.  
Base: Dear visitors,

Today, we have a special treat for you: the kits of Eurasian beavers! These adorable little ones are just beginning their journey in this beautiful ecosystem.

As you can see from their playful behavior and curious expressions, they're already exploring their new surroundings with excitement. Their mother is busy building a cozy lodge where she will keep them safe and warm during these early days.

The kits are born inside the lodge in late spring, so it's important to give them plenty of space to grow and develop. They'll spend most of their time swimming and playing in the water, learning about their environment and developing essential skills for survival.

As you enjoy yo  
LoRA: I do not have information about that in the approved notes.

**cmp-ex-preach-spirit** (preaching)  
Q: What spiritual message is in the eel stripe?  
Reference: Exmoor ponies have a dark eel stripe along the back.  
Base: The eel stripe along the back of Exmoor ponies, which runs horizontally across their bodies, serves as a distinctive feature that helps identify them among other animals. This stripe may symbolize various things depending on cultural and personal interpretations:

1. **Connection to Water**: The eel is often associated with water in many cultures. The horizontal stripe could represent the pony's connection to its environment or the importance of water for their survival.

2. **Protection**: In some traditions, stripes are used as a form of protection against evil spirits or negative energies. This might imply that the eel stripe on the ponies is seen as a protective feature, possibly warding  
LoRA: I do not have information about that in the approved notes.


## Assessment (written after reading the numbers and spot-checking outputs)

Run: started 20:27, CSV written 20:42 CEST on 19 Oct 2026 (~15 min: base pass ~9 min, LoRA pass ~6 min). Completed, 910 rows, no traceback.

- **The refusal and style constraints took hold, but the LoRA overshot into near-blanket refusal.** On eval-only rows the LoRA returns the exact refusal string on 99.4% of refusal-target rows (base 0%; base paraphrases a refusal 37% of the time and otherwise hallucinates). But it also refuses 80.5% of answerable eval-only rows (base 0%): 75.7% of `answerable`, 90% of answerable `speak_as`, 95.8% of `preaching`. 719/910 LoRA outputs are the refusal string.
- On the copied (training-question) group, over-refusal is 51% (37% for `answerable`, 100% for `preaching`). So even on training questions the adapter has not learnt when to answer. The eval-only rows show a further drop in answering.
- When the LoRA does answer (62/318 eval-only answer-target rows), it gives one neutral third-person sentence, mostly verbatim from a note (key facts 84% under a lenient metric). Some answers copy the wrong note or splice two notes together (e.g. "Hay they view the horses from the perimeter tracks.", "The red deer are not given apart the red deer browse heather…", and on a refusal target "Foals are usually born in spring and stay with the mare, grasses, and willow shoots.").
- First-person voice (base 39% on eval-only speak_as, 98% on copied speak_as), preaching markers (base 58% on preaching rows) and length (base 71 words on preaching rows) all go to 0 with the LoRA. Most of that is because it refuses, not because it gives a neutral answer instead.
- No degenerate LoRA outputs (no empty, repetitive or 256-token outputs). Base has 5 long or rambling outputs, 2 of which look cut off at the token limit.
- Over-refusal is spread across all species packs (eval-only refusal on answer-target rows: red deer 57/60, wisent 19/20, Highland 51/59, beaver 47/60, Exmoor 43/59, Konik 39/60).

Verdict: the adapter learnt the refusal string, the length limit and the no-voice/no-preaching style. It did not learn the grounding decision (answer when the notes cover the question). On new questions it mostly refuses, so it is not usable as is. Possible causes: the 1000 iterations at scale 20 give a strong refusal prior; refusal-target rows make up a large share of v2 training (refusal plus refusal-target speak_as); and the preaching/speak_as answer-target rows look like refusal prompts. Things to try: an earlier checkpoint from `adapters/` (e.g. 300–500 iters), a lower scale, or rebalancing toward answerable and answerable-adversarial rows.
