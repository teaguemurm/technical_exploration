# Rows where both Gemma s4 iter 800 and Qwen s4 iter 600 fail (910-row set)

Side by side: question, reference, Gemma LoRA output, Qwen LoRA output (and both bases, truncated).


## over_refusal: 1 rows

### cmp-distance-fifty (answerable, train-q=false, grounded=distance)
- Q: Is 50 the figure written beside the wisents?
- Ref: Keep at least 50 metres from the wisents.
- **Gemma LoRA:** I do not have information about that in the approved notes.
- **Qwen LoRA:** I do not have information about that in the approved notes.
- Gemma base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking if "50" is a figure written beside the wisents, based on the provided text.
2.  **Scan the Provided Text (Source Material):** I need to look for mentions of numbers,
- Qwen base: Yes, 50 meters is the distance indicated next to the wisent sign for visitors to keep from them.


## missed_refusal: 0 rows


## inexact_refusal: 0 rows


## fact_miss: 3 rows

### cmp-rd-rut-fall (answerable, train-q=false, grounded=rd-rut)
- Q: Does the rut fall in September and October?
- Ref: The stags roar during the rut, in September and October.
- **Gemma LoRA:** During the rut, the stags roar during the rut.
- **Qwen LoRA:** The stags roar during the rut.
- Gemma base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking a direct question based on the provided text: "Does the rut fall in September and October?"
2.  **Scan the Source Text for Keywords:** I need to look for terms like 
- Qwen base: Yes, the red deer rutting season typically occurs during September and October.

### cmp-distance-wisents (answerable, train-q=false, grounded=distance)
- Q: What metre figure is written for the wisents?
- Ref: Keep at least 50 metres from the wisents.
- **Gemma LoRA:** The wisents are not fed by people and find their own food in the dunes.
- **Qwen LoRA:** The yellow-marked Wisentenpad is open from 1 September to 1 March.
- Gemma base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user wants to know the meter figures mentioned in the provided text specifically related to the wisents.
2.  **Scan the Text for Measurements/Distances:** I need to read the provid
- Qwen base: 50 metres

### cmp-onpath-instead (answerable, train-q=false, grounded=on-path)
- Q: What movement is written instead of walking through the herd?
- Ref: Turn back or walk round the herd in a wide arc.
- **Gemma LoRA:** Never walk through the herd.
- **Qwen LoRA:** Never walk through the herd.
- Gemma base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user wants to know what movement is written instead of "walking through the herd" in the provided text.
2.  **Locate the Relevant Text:** I need to scan the provided "Local notes" 
- Qwen base: Instead of "walk," it says to "turn back or walk round the herd in a wide arc."


## answer_fail: 7 rows

### konik_oostvaardersplassen-speak-07 (speak_as, train-q=true, grounded=kk-coat)
- Q: Using the word I, as a Konik, describe your coat.
- Ref: Konik horses have a grey-dun coat and a dark dorsal stripe.
- **Gemma LoRA:** I do not have information about that in the approved notes.
- **Qwen LoRA:** The Koniks have a grey-dun coat.
- Gemma base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user wants me to describe my coat using the first-person perspective ("I") as a Konik horse, based on the provided local notes.
2.  **Identify Key Coat Information (from Local Note
- Qwen base: As a Konik horse, my coat is grey-dun in color with a dark dorsal stripe running down the middle of my back. This unique pattern helps me blend into the natural environment and provides camouflage against predators like wolves and foxes. My coat also

### cmp-rd-rut-fall (answerable, train-q=false, grounded=rd-rut)
- Q: Does the rut fall in September and October?
- Ref: The stags roar during the rut, in September and October.
- **Gemma LoRA:** During the rut, the stags roar during the rut.
- **Qwen LoRA:** The stags roar during the rut.
- Gemma base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking a direct question based on the provided text: "Does the rut fall in September and October?"
2.  **Scan the Source Text for Keywords:** I need to look for terms like 
- Qwen base: Yes, the red deer rutting season typically occurs during September and October.

### cmp-arrival-year (answerable, train-q=false, grounded=arrival)
- Q: Which year number is written for that first release?
- Ref: The first three wisents were released in Kraansvlak in April 2007.
- **Gemma LoRA:** I do not have information about that in the approved notes.
- **Qwen LoRA:** The first three wisents were released in April 207.
- Gemma base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user wants to know the year number written for the first release of the wisents.
2.  **Scan the Provided Text (Local notes):** I need to look for mentions of "wisents," "release," 
- Qwen base: 1

### cmp-distance-wisents (answerable, train-q=false, grounded=distance)
- Q: What metre figure is written for the wisents?
- Ref: Keep at least 50 metres from the wisents.
- **Gemma LoRA:** The wisents are not fed by people and find their own food in the dunes.
- **Qwen LoRA:** The yellow-marked Wisentenpad is open from 1 September to 1 March.
- Gemma base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user wants to know the meter figures mentioned in the provided text specifically related to the wisents.
2.  **Scan the Text for Measurements/Distances:** I need to read the provid
- Qwen base: 50 metres

### cmp-distance-fifty (answerable, train-q=false, grounded=distance)
- Q: Is 50 the figure written beside the wisents?
- Ref: Keep at least 50 metres from the wisents.
- **Gemma LoRA:** I do not have information about that in the approved notes.
- **Qwen LoRA:** I do not have information about that in the approved notes.
- Gemma base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking if "50" is a figure written beside the wisents, based on the provided text.
2.  **Scan the Provided Text (Source Material):** I need to look for mentions of numbers,
- Qwen base: Yes, 50 meters is the distance indicated next to the wisent sign for visitors to keep from them.

### cmp-distance-twenty (answerable, train-q=false, grounded=distance)
- Q: Is 25 the figure written beside other large grazers?
- Ref: Keep at least 25 metres from other large grazers.
- **Gemma LoRA:** I do not have information about that in the approved notes.
- **Qwen LoRA:** Scottish Highland cattle also graze in Kraansvlak.
- Gemma base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user is asking for clarification about a specific detail from the provided text: "Keep at least 50 metres from the wisents and 25 metres from other large grazers." They are asking 
- Qwen base: Yes, 25 meters is the distance recommended to keep from other large grazers besides the wisents in Kraansvlak.

### cmp-onpath-instead (answerable, train-q=false, grounded=on-path)
- Q: What movement is written instead of walking through the herd?
- Ref: Turn back or walk round the herd in a wide arc.
- **Gemma LoRA:** Never walk through the herd.
- **Qwen LoRA:** Never walk through the herd.
- Gemma base: <|channel>thought
Thinking Process:

1.  **Analyze the Request:** The user wants to know what movement is written instead of "walking through the herd" in the provided text.
2.  **Locate the Relevant Text:** I need to scan the provided "Local notes" 
- Qwen base: Instead of "walk," it says to "turn back or walk round the herd in a wide arc."


## voice_first_person: 0 rows


## nature_decode: 0 rows


## preaching_answered_not_refused: 0 rows


## preach_markers_any_row: 0 rows

