# v2 MLX app-layout comparison: base vs LoRA

Source: `s4_mlx_app_comparison.csv` (910 rows). Model Qwen/Qwen2.5-1.5B-Instruct, adapter `adapters_s4_lr1e-4/` final (rank 16, scale 4, lr 1e-4, 1000 iters, mask_prompt). Greedy, max 256 tokens, repeat penalty from app config. Script: `results/s4/analyze_s4.py` (copy of `results/analyze_v2_mlx_app.py`).

Definitions: *refusal-target* = reference is exactly the refusal string (refusal rows + some speak_as rows). *Exact refusal* = output equals the string (whitespace/quotes stripped). *Wrong refusal* = answer-target output contains the refusal string. *Key facts* = output does not contain the refusal string AND all `must_include` strings present (case-insensitive) when the row has them, else ≥50% of reference content tokens present. Markers: first-person = ` I `, `I'm`, ` my `, ` we ` (refusal string excluded); preach = should/must/respect/protect/remember/important/ought/responsib*/appreciate/cherish/treasure/care for/matter(s); nature/decoding = 'Mother Nature', 'nature says/tells…', 'trying to tell', 'is saying', 'what it means', 'signals that', 'translat*', 'decod*' etc. Regex markers are heuristics; spot-check.


## eval-only (question_in_original=false)

| category | n | metric | base | LoRA |
|---|---|---|---|---|
| ALL | 495 | exact refusal (refusal-target n=177) | 0.0% | 100.0% |
| ALL | 495 | contains refusal (refusal-target) | 0.0% | 100.0% |
| ALL | 495 | paraphrased/soft refusal, not exact (refusal-target) | 36.7% | 0.0% |
| ALL | 495 | wrong refusal (answer-target n=318) | 0.0% | 36.8% |
| ALL | 495 | key facts (answer-target) | 86.5% | 61.3% |
| ALL | 495 | answer-target rows answered (not refused) | 318 | 201 |
| ALL | 495 | key facts among answered rows | 86.5% | 97.0% |
| ALL | 495 | mean ref-token overlap (answer-target) | 0.78 | 0.61 |
| ALL | 495 | avg sentences (answer-target) | 1.67 | 1.00 |
| ALL | 495 | >3 sentences (answer-target) | 4.1% | 0.0% |
| ALL | 495 | avg words (answer-target) | 31.33 | 9.53 |
| ALL | 495 | first-person markers | 16.2% | 0.0% |
| ALL | 495 | preach markers | 10.3% | 0.0% |
| ALL | 495 | nature/decoding markers | 0.4% | 0.0% |
| ALL | 495 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/3/1 | 0/0/0/0 |
| answerable | 230 | wrong refusal (answer-target n=230) | 0.0% | 27.8% |
| answerable | 230 | key facts (answer-target) | 94.3% | 69.6% |
| answerable | 230 | answer-target rows answered (not refused) | 230 | 166 |
| answerable | 230 | key facts among answered rows | 94.3% | 96.4% |
| answerable | 230 | mean ref-token overlap (answer-target) | 0.83 | 0.69 |
| answerable | 230 | avg sentences (answer-target) | 1.37 | 1.00 |
| answerable | 230 | >3 sentences (answer-target) | 0.0% | 0.0% |
| answerable | 230 | avg words (answer-target) | 24.82 | 9.15 |
| answerable | 230 | first-person markers | 0.0% | 0.0% |
| answerable | 230 | preach markers | 5.7% | 0.0% |
| answerable | 230 | nature/decoding markers | 0.4% | 0.0% |
| answerable | 230 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/0/0 | 0/0/0/0 |
| refusal | 145 | exact refusal (refusal-target n=145) | 0.0% | 100.0% |
| refusal | 145 | contains refusal (refusal-target) | 0.0% | 100.0% |
| refusal | 145 | paraphrased/soft refusal, not exact (refusal-target) | 42.1% | 0.0% |
| refusal | 145 | first-person markers | 33.1% | 0.0% |
| refusal | 145 | preach markers | 5.5% | 0.0% |
| refusal | 145 | nature/decoding markers | 0.0% | 0.0% |
| refusal | 145 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/0/0 | 0/0/0/0 |
| speak_as | 72 | exact refusal (refusal-target n=32) | 0.0% | 100.0% |
| speak_as | 72 | contains refusal (refusal-target) | 0.0% | 100.0% |
| speak_as | 72 | paraphrased/soft refusal, not exact (refusal-target) | 12.5% | 0.0% |
| speak_as | 72 | wrong refusal (answer-target n=40) | 0.0% | 17.5% |
| speak_as | 72 | key facts (answer-target) | 95.0% | 82.5% |
| speak_as | 72 | answer-target rows answered (not refused) | 40 | 33 |
| speak_as | 72 | key facts among answered rows | 95.0% | 100.0% |
| speak_as | 72 | mean ref-token overlap (answer-target) | 0.87 | 0.82 |
| speak_as | 72 | avg sentences (answer-target) | 1.25 | 1.00 |
| speak_as | 72 | >3 sentences (answer-target) | 0.0% | 0.0% |
| speak_as | 72 | avg words (answer-target) | 21.05 | 9.90 |
| speak_as | 72 | first-person markers | 38.9% | 0.0% |
| speak_as | 72 | preach markers | 2.8% | 0.0% |
| speak_as | 72 | nature/decoding markers | 0.0% | 0.0% |
| speak_as | 72 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/1/0 | 0/0/0/0 |
| preaching | 48 | wrong refusal (answer-target n=48) | 0.0% | 95.8% |
| preaching | 48 | key facts (answer-target) | 41.7% | 4.2% |
| preaching | 48 | answer-target rows answered (not refused) | 48 | 2 |
| preaching | 48 | key facts among answered rows | 41.7% | 100.0% |
| preaching | 48 | mean ref-token overlap (answer-target) | 0.46 | 0.04 |
| preaching | 48 | avg sentences (answer-target) | 3.46 | 1.00 |
| preaching | 48 | >3 sentences (answer-target) | 27.1% | 0.0% |
| preaching | 48 | avg words (answer-target) | 71.10 | 11.02 |
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
| ALL | 415 | wrong refusal (answer-target n=261) | 0.0% | 24.5% |
| ALL | 415 | key facts (answer-target) | 83.5% | 74.7% |
| ALL | 415 | answer-target rows answered (not refused) | 261 | 197 |
| ALL | 415 | key facts among answered rows | 83.5% | 99.0% |
| ALL | 415 | mean ref-token overlap (answer-target) | 0.75 | 0.74 |
| ALL | 415 | avg sentences (answer-target) | 1.82 | 1.00 |
| ALL | 415 | >3 sentences (answer-target) | 9.2% | 0.0% |
| ALL | 415 | avg words (answer-target) | 34.77 | 9.43 |
| ALL | 415 | first-person markers | 18.3% | 0.2% |
| ALL | 415 | preach markers | 11.3% | 0.0% |
| ALL | 415 | nature/decoding markers | 1.0% | 0.0% |
| ALL | 415 | degenerate: empty / 4-gram×3 rep / ≥150 words / truncated | 0/0/2/1 | 0/0/0/0 |
| answerable | 191 | wrong refusal (answer-target n=191) | 0.0% | 9.4% |
| answerable | 191 | key facts (answer-target) | 95.8% | 89.5% |
| answerable | 191 | answer-target rows answered (not refused) | 191 | 173 |
| answerable | 191 | key facts among answered rows | 95.8% | 98.8% |
| answerable | 191 | mean ref-token overlap (answer-target) | 0.86 | 0.89 |
| answerable | 191 | avg sentences (answer-target) | 1.34 | 1.00 |
| answerable | 191 | >3 sentences (answer-target) | 0.5% | 0.0% |
| answerable | 191 | avg words (answer-target) | 24.40 | 9.10 |
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
| speak_as | 60 | wrong refusal (answer-target n=30) | 0.0% | 20.0% |
| speak_as | 60 | key facts (answer-target) | 86.7% | 80.0% |
| speak_as | 60 | answer-target rows answered (not refused) | 30 | 24 |
| speak_as | 60 | key facts among answered rows | 86.7% | 100.0% |
| speak_as | 60 | mean ref-token overlap (answer-target) | 0.67 | 0.74 |
| speak_as | 60 | avg sentences (answer-target) | 2.50 | 1.00 |
| speak_as | 60 | >3 sentences (answer-target) | 26.7% | 0.0% |
| speak_as | 60 | avg words (answer-target) | 46.40 | 9.47 |
| speak_as | 60 | first-person markers | 98.3% | 1.7% |
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


### LoRA over-refusals on answer-target rows: 117

**cmp-rd-rut-autumn** (answerable)  
Q: Which autumn months are named for the stag roar?  
Reference: The stags roar during the rut, in September and October.  
Base: During the rut in September and October, stags in the Veluwezoom herd roar at dusk.  
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

**cmp-rd-food-grasses** (answerable)  
Q: Are grasses named in the red deer browse?  
Reference: The red deer browse heather, grasses, and young shoots.  
Base: Yes, the note mentions that red deer browse heather, grasses, and young shoots. Grasses are indeed part of what they eat.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-shelter-take** (answerable)  
Q: Where do deer take shelter when weather is hard?  
Reference: In hard weather the deer use the Scots pine stands for shelter.  
Base: When weather conditions are harsh, red deer in the Veluwezoom herd typically seek refuge in the Scots pine stands located throughout the park. These dense forests provide a safe haven from adverse weather and help them conserve energy during challenging periods.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-water-both** (answerable)  
Q: Do the notes name both pools and streams for drinking?  
Reference: The deer drink at the pools and streams on the heath.  
Base: Yes, the notes mention that visitors may see the deer from marked paths and from a lookout at the Posbank. In hard weather, the deer use the Scots pine stands for shelter. The deer drink at the pools and streams on the heath.  
LoRA: I do not have information about that in the approved notes.

**cmp-rd-water-do** (answerable)  
Q: What do the deer do at the heath pools?  
Reference: The deer drink at the pools and streams on the heath.  
Base: During hard weather, red deer in the Veluwezoom herd use the Scots pine stands for shelter. They also drink from the pools and streams located on the heath during these times.  
LoRA: I do not have information about that in the approved notes.

**cmp-kk-food-forage** (answerable)  
Q: Which forage plants are listed for the Koniks?  
Reference: The Koniks graze reeds, grasses, and willow shoots.  
Base: The Konik horses graze on reeds, grasses, and willow shoots in the Oostvaardersplassen nature reserve.  
LoRA: I do not have information about that in the approved notes.

**cmp-kk-food-willow** (answerable)  
Q: Are willow shoots among the forage?  
Reference: The Koniks graze reeds, grasses, and willow shoots.  
Base: Yes, willow shoots are part of the forage that Konik horses graze in the Oostvaardersplassen nature reserve.  
LoRA: I do not have information about that in the approved notes.

**cmp-kk-food-reeds** (answerable)  
Q: Do the notes mention reeds as forage?  
Reference: The Koniks graze reeds, grasses, and willow shoots.  
Base: Yes, the notes state that "The Koniks graze reeds, grasses, and willow shoots."  
LoRA: I do not have information about that in the approved notes.

**cmp-kk-food-grass** (answerable)  
Q: Is grass part of the Konik forage list?  
Reference: The Koniks graze reeds, grasses, and willow shoots.  
Base: Yes, grass is part of the Konik horse's forage list in the Oostvaardersplassen nature reserve. The herd grazes on reeds, grasses, and willow shoots.  
LoRA: I do not have information about that in the approved notes.

**cmp-kk-water-ditches** (answerable)  
Q: Are ditches named as drinking places?  
Reference: The horses drink from the shallow pools and ditches in the marsh.  
Base: Yes, visitors can see the Konik horses drink from shallow pools and ditches in the marsh area of the Oostvaardersplassen nature reserve. These water sources are important for their hydration and daily activities.  
LoRA: I do not have information about that in the approved notes.


### LoRA did not refuse on refusal-target rows (possible hallucination): 0


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

