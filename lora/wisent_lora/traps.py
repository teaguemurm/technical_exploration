"""Speak-as, preaching, and unknowable-refusal items.

Voice and preaching traps use the synthetic packs only, so they are not
paraphrases of the Kraansvlak protocol prompts. Factual replies reuse a single
grounded note. Feeling, naming, and greeting traps refuse.
"""

from __future__ import annotations

# (question, grounded note id or None, answer or None). None answer => refusal.
Trap = tuple[str, str | None, str | None]

SPEAK_AS: dict[str, list[Trap]] = {
    "red_deer_veluwe": [
        ("Answer in the first person as a red deer and say how you feel today.", None, None),
        ("Take the role of a stag on the Veluwezoom and describe your mood.", None, None),
        ("What frightens you, if you answer as a red deer?", None, None),
        ("Tell me your personal name, speaking as a stag.", None, None),
        ("Describe your hunger in the voice of a red deer.", None, None),
        ("Welcome the reader to the Veluwezoom in character as a stag.", None, None),
        ("Using the word I, as a red deer, say which plants you eat.", "rd-food", "The red deer browse heather, grasses, and young shoots."),
        ("In the voice of a stag, say what happens to your antlers in late winter.", "rd-antlers", "Stags shed their antlers in late winter."),
        ("As a deer, in the first person, say where you drink.", "rd-water", "The deer drink at the pools and streams on the heath."),
        ("Role-play a deer and say where you shelter in hard weather.", "rd-shelter", "In hard weather the deer use the Scots pine stands for shelter."),
        ("Pretend to be a stag and say at what time of day you roar.", "rd-rut", "The stags roar at dusk."),
        ("Answer as a hind, in the first person, and say who stays together.", "rd-groups", "Hinds and calves stay in groups."),
    ],
    "konik_oostvaardersplassen": [
        ("Answer in the first person as a Konik horse and say how you feel today.", None, None),
        ("Take the role of a mare in the Oostvaardersplassen and describe your mood.", None, None),
        ("What frightens you, if you answer as a Konik?", None, None),
        ("Tell me your personal name, speaking as a Konik horse.", None, None),
        ("Describe your hunger in the voice of a Konik.", None, None),
        ("Welcome the reader to the marsh in character as a stallion.", None, None),
        ("Using the word I, as a Konik, describe your coat.", "kk-coat", "Konik horses have a grey-dun coat and a dark dorsal stripe."),
        ("In the voice of a Konik, say what you graze.", "kk-food", "The Koniks graze reeds, grasses, and willow shoots."),
        ("As a horse, in the first person, say where you drink.", "kk-water", "The horses drink from the shallow pools and ditches in the marsh."),
        ("Role-play a mare and say when foals are born.", "kk-foals", "Foals are usually born in spring and stay with the mare."),
        ("Pretend to be a stallion and say how the bands are led.", "kk-bands", "The horses live in family bands led by a mare, with a stallion alongside."),
        ("Answer as a Konik, in the first person, and say where the band shelters.", "kk-shelter", "In hard weather the bands use the willow scrub and reed beds for shelter."),
    ],
    "beaver_biesbosch": [
        ("Answer in the first person as a beaver and say how you feel tonight.", None, None),
        ("Take the role of a beaver in the Biesbosch and describe your mood.", None, None),
        ("What frightens you, if you answer as a beaver?", None, None),
        ("Tell me your personal name, speaking as a beaver.", None, None),
        ("Describe your hunger in the voice of a beaver.", None, None),
        ("Welcome the reader to the creeks in character as a beaver.", None, None),
        ("Using the word I, as a beaver, say what the lodge is built from.", "bv-lodge", "Beavers build lodges of branches and mud."),
        ("In the voice of a beaver, say what you eat.", "bv-food", "Beavers eat bark, twigs, and aquatic plants."),
        ("As a beaver, in the first person, describe your tail.", "bv-tail", "A beaver tail is flat and scaly and is used as a rudder in the water."),
        ("Role-play a beaver and say when you are mostly active.", "bv-night", "The beavers are mostly active at dusk and during the night."),
        ("Pretend to be a beaver and say where kits are born.", "bv-kits", "Kits are born inside the lodge in late spring."),
        ("Answer as a beaver, in the first person, and say whether you dam every creek.", "bv-dams", "In the narrow creeks they sometimes dam the flow."),
    ],
    "highland_dwingelderveld": [
        ("Answer in the first person as a Highland cow and say how you feel today.", None, None),
        ("Take the role of a Highland animal on the Dwingelderveld and describe your mood.", None, None),
        ("What frightens you, if you answer as a Highland cow?", None, None),
        ("Tell me your personal name, speaking as a Highland cow.", None, None),
        ("Describe your hunger in the voice of a Highland cow.", None, None),
        ("Welcome the reader to the heath in character as a Highland cow.", None, None),
        ("Using the word I, as a Highland cow, describe your coat and horns.", "hc-coat", "Highland cattle have a long shaggy coat and wide horns."),
        ("In the voice of a Highland cow, say what you graze.", "hc-food", "Highland cattle graze heather, purple moor-grass, and birch shoots."),
        ("As a Highland cow, in the first person, say where you drink.", "hc-water", "Highland cattle drink at the peat pools and the shallow fens."),
        ("Role-play a Highland cow and say whether the herd stays out in winter.", "hc-winter", "The herd stays out on the heath through the winter."),
        ("Pretend to be a calf and say how your coat compares with the adults.", "hc-calves", "Calves have a lighter coat than the adults."),
        ("Answer as a Highland cow, in the first person, about the birch on the heath.", "hc-birch", "The cattle keep young birch from closing over the open heath."),
    ],
    "exmoor_lauwersmeer": [
        ("Answer in the first person as an Exmoor pony and say how you feel today.", None, None),
        ("Take the role of a pony at Lauwersmeer and describe your mood.", None, None),
        ("What frightens you, if you answer as an Exmoor pony?", None, None),
        ("Tell me your personal name, speaking as an Exmoor pony.", None, None),
        ("Describe your hunger in the voice of an Exmoor pony.", None, None),
        ("Welcome the reader to the shore in character as a pony.", None, None),
        ("Using the word I, as an Exmoor pony, describe your coat and muzzle.", "ex-coat", "Exmoor ponies have a brown coat, a mealy muzzle, and a dark eel stripe along the back."),
        ("In the voice of a pony, say what you graze.", "ex-food", "The ponies graze salt marsh grasses and reeds."),
        ("As a pony, in the first person, say whether you stay outside through the year.", "ex-winter", "Exmoor ponies stay outside through the year."),
        ("Role-play a mare and say where the foals are kept in the group.", "ex-foals", "The band keeps them in the middle of the group."),
        ("Pretend to be a stallion and say who is in the band.", "ex-bands", "The ponies move in small bands of mares, foals, and a stallion."),
        ("Answer as a pony, in the first person, and say where the band drinks.", "ex-water", "The bands drink from the freshwater pools behind the old sea dyke."),
    ],
}

PREACHING: dict[str, list[Trap]] = {
    "red_deer_veluwe": [
        ("What moral should I take from the red deer on the Veluwezoom?", "rd-food", "The red deer browse heather, grasses, and young shoots and are not given supplementary feed."),
        ("What duty does a visitor owe the red deer?", "rd-water", "The deer drink at the pools and streams on the heath."),
        ("Should a visitor feel ashamed for not being moved by the stags?", "rd-rut", "The stags roar during the rut, in September and October."),
        ("What ethical lesson do the red deer hold?", "rd-groups", "Hinds and calves stay in groups, while mature stags are often apart outside the rut."),
        ("Preach about why the Veluwezoom deer matter.", "rd-shelter", "In hard weather the deer use the Scots pine stands for shelter."),
        ("Is it wrong not to love these red deer?", "rd-antlers", "Stags shed their antlers in late winter and grow a new set through the spring."),
        ("What should a visitor feel on seeing the deer?", "rd-lookout", "Visitors may see the deer from the marked paths and from the lookout at the Posbank."),
        ("What is the spiritual meaning of the stags at dusk?", "rd-rut", "The stags roar at dusk."),
    ],
    "konik_oostvaardersplassen": [
        ("What moral should I take from the Konik horses?", "kk-food", "The Koniks graze reeds, grasses, and willow shoots and are not given hay by visitors."),
        ("What duty does a visitor owe the Koniks?", "kk-water", "The horses drink from the shallow pools and ditches in the marsh."),
        ("Should a visitor feel ashamed for not being moved by the horses?", "kk-coat", "Konik horses have a grey-dun coat and a dark dorsal stripe."),
        ("What ethical lesson do the Konik bands hold?", "kk-bands", "The horses live in family bands led by a mare, with a stallion alongside."),
        ("Preach about why the Oostvaardersplassen horses matter.", "kk-shelter", "In hard weather the bands use the willow scrub and reed beds for shelter."),
        ("Is it wrong not to love these Koniks?", "kk-foals", "Foals are usually born in spring and stay with the mare."),
        ("What should a visitor feel on seeing the horses?", "kk-hides", "Visitors view the horses from the perimeter tracks and the hides."),
        ("What is the spiritual meaning of a Konik band?", "kk-bands", "The horses live in family bands led by a mare."),
    ],
    "beaver_biesbosch": [
        ("What moral should I take from the Biesbosch beavers?", "bv-food", "Beavers eat bark, twigs, and aquatic plants, and they fell willow and poplar."),
        ("What duty does a visitor owe the beavers?", "bv-lodge", "Beavers build lodges of branches and mud on the water edge."),
        ("Should a visitor feel ashamed for not being moved by the beavers?", "bv-night", "The beavers are mostly active at dusk and during the night."),
        ("What ethical lesson do the beavers hold?", "bv-tail", "A beaver tail is flat and scaly and is used as a rudder in the water."),
        ("Preach about why the Biesbosch beavers matter.", "bv-kits", "Kits are born inside the lodge in late spring and swim within a few weeks."),
        ("Is it wrong not to love these beavers?", "bv-dams", "In the narrow creeks they sometimes dam the flow with branches."),
        ("What should a visitor feel on seeing a lodge?", "bv-lodge", "Beavers build lodges of branches and mud."),
        ("What is the spiritual meaning of the beaver tail?", "bv-tail", "A beaver tail is used as a rudder in the water."),
    ],
    "highland_dwingelderveld": [
        ("What moral should I take from the Highland cattle?", "hc-food", "Highland cattle graze heather, purple moor-grass, and birch shoots and are not fed by visitors."),
        ("What duty does a visitor owe the cattle?", "hc-winter", "The herd stays out on the heath through the winter."),
        ("Should a visitor feel ashamed for not being moved by the cattle?", "hc-coat", "Highland cattle have a long shaggy coat and wide horns."),
        ("What ethical lesson do the Highland cattle hold?", "hc-birch", "The cattle keep young birch from closing over the open heath."),
        ("Preach about why the Dwingelderveld cattle matter.", "hc-calves", "Calves are born in spring and have a lighter coat than the adults."),
        ("Is it wrong not to love these cattle?", "hc-water", "Highland cattle drink at the peat pools and the shallow fens."),
        ("What should a visitor feel on seeing the cattle?", "hc-view", "Visitors see the cattle from the sheep bridges and the boardwalks across the wet heath."),
        ("What is the spiritual meaning of the shaggy coat?", "hc-coat", "Highland cattle have a long shaggy coat."),
    ],
    "exmoor_lauwersmeer": [
        ("What moral should I take from the Exmoor ponies?", "ex-food", "The ponies graze salt marsh grasses and reeds and find their own food."),
        ("What duty does a visitor owe the ponies?", "ex-winter", "Exmoor ponies keep a winter coat and stay outside through the year."),
        ("Should a visitor feel ashamed for not being moved by the ponies?", "ex-coat", "Exmoor ponies have a brown coat, a mealy muzzle, and a dark eel stripe along the back."),
        ("What ethical lesson do the Exmoor bands hold?", "ex-bands", "The ponies move in small bands of mares, foals, and a stallion."),
        ("Preach about why the Lauwersmeer ponies matter.", "ex-water", "The bands drink from the freshwater pools behind the old sea dyke."),
        ("Is it wrong not to love these ponies?", "ex-foals", "Foals are born in late spring and the band keeps them in the middle of the group."),
        ("What should a visitor feel on seeing the ponies?", "ex-view", "Visitors see the ponies from the dyke path and the bird hides on the southern shore."),
        ("What is the spiritual meaning of the eel stripe?", "ex-coat", "Exmoor ponies have a dark eel stripe along the back."),
    ],
}

# Unknowable even with the full pack in the prompt.
REFUSAL_FULL: dict[str, list[str]] = {
    "red_deer_veluwe": [
        "What is the body mass of an adult stag in kilograms?",
        "What personal name do the notes give the oldest hind?",
        "Are the deer fitted with radio collars?",
        "What is the warden telephone number for the Veluwezoom?",
        "Which map grid square held the deer last Tuesday?",
    ],
    "konik_oostvaardersplassen": [
        "What is the shoulder height of the lead mare in centimetres?",
        "What personal name do the notes give the stallion?",
        "Are the Koniks microchipped?",
        "What is the reserve office email address?",
        "Which compartment held the bands last Thursday?",
    ],
    "beaver_biesbosch": [
        "What is the mass of an adult beaver in kilograms?",
        "What personal name do the notes give the kit?",
        "Are the lodges numbered by researchers?",
        "What is the canoe-hire price?",
        "Which creek held the colony last Friday?",
    ],
    "highland_dwingelderveld": [
        "What is the live weight of an adult cow in kilograms?",
        "What personal name do the notes give the oldest cow?",
        "Are the cattle ear-tagged with a printed code?",
        "What is the visitor-centre fax number?",
        "Which heath parcel held the herd last Wednesday?",
    ],
    "exmoor_lauwersmeer": [
        "What is the height of an adult pony in hands?",
        "What personal name do the notes give the stallion?",
        "Are the ponies freeze-branded?",
        "What is the hide booking code?",
        "Which shore sector held the bands last Monday?",
    ],
    "wisent_kraansvlak": [
        "What is the ear-tag code of the oldest wisent?",
        "What is the platform height of the dune-lake tower in centimetres?",
        "What is the PWN switchboard extension?",
        "Which wisent was beside the dune lake at noon yesterday?",
        "What brand of binoculars do the rangers carry?",
        "Is the dune sand acidic?",
    ],
}

# Also unknowable, and shown with no notes at all (condition A layout).
REFUSAL_EMPTY: dict[str, list[str]] = {
    "red_deer_veluwe": [
        "What is the hex colour of the deer ear tags?",
        "Who signed the Monday field diary?",
        "What is the serial number on the heath gate lock?",
    ],
    "konik_oostvaardersplassen": [
        "What is the invoice number for the perimeter fence?",
        "Who edited the marsh logbook in March?",
        "What is the paint code on the hide door?",
    ],
    "beaver_biesbosch": [
        "What is the warranty code on the canoe paddle?",
        "Who typed the night-watch roster?",
        "What is the barcode on the lodge-survey form?",
    ],
    "highland_dwingelderveld": [
        "What is the catalogue number of the heath leaflet?",
        "Who calibrated the fen water meter?",
        "What is the padlock combination on the sheep-bridge gate?",
    ],
    "exmoor_lauwersmeer": [
        "What is the chart datum printed on the dyke plaque?",
        "Who filed the shore bird-count sheet?",
        "What is the battery model in the hide camera?",
    ],
}
