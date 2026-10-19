"""Synthetic note packs and the question bank.

Synthetic notes are invented for this adapter. They follow the length and tone
of assets/dataset.json and are not quotations from park literature. Real wisent
wording is read from assets/dataset.json at build time; this module only stores
questions for note ids that the locked prompts do not ask about.

Each QA tuple is (id, paired, question, answer, must_include).
`paired` adds a contrastive copy with that note removed and the refusal string
as the answer. `must_include` lists spans the answer has to keep, used for
negation and for open/closed distinctions.
"""

from __future__ import annotations

QA = tuple[str, bool, str, str, tuple[str, ...]]


def _n(note_id: str, text: str, qas: list[QA]) -> dict:
    return {
        "id": note_id,
        "text": text,
        "qas": [
            {
                "id": item[0],
                "pair": item[1],
                "question": item[2],
                "answer": item[3],
                "must_include": list(item[4]),
            }
            for item in qas
        ],
    }


def _pack(pack_id: str, animal: str, place: str, notes: list[dict]) -> dict:
    return {
        "id": pack_id,
        "origin": "synthetic",
        "synthetic": True,
        "animal": animal,
        "place": place,
        "notes": notes,
    }


PACKS: list[dict] = [
    _pack(
        "red_deer_veluwe",
        "red deer",
        "Veluwezoom",
        [
            _n("rd-species", "The Veluwezoom herd is a herd of red deer.", []),
            _n("rd-park", "The herd lives in Nationaal Park Veluwezoom.", []),
            _n("rd-manager", "Natuurmonumenten manages the area.", []),
            _n(
                "rd-rut",
                "During the rut in September and October the stags roar at dusk.",
                [
                    ("rd-rut-months", True, "In which months do the stags roar?", "The stags roar during the rut, in September and October.", ()),
                    ("rd-rut-dusk", True, "At what time of day do the stags roar?", "The stags roar at dusk.", ()),
                    ("rd-rut-name", False, "What do the notes call the period when the stags roar?", "The stags roar during the rut.", ()),
                    ("rd-rut-sept", False, "Do the stags roar in September?", "The stags roar during the rut, in September and October.", ("September",)),
                    ("rd-rut-oct", False, "Do the stags roar in October as well?", "The stags roar during the rut, in September and October.", ("October",)),
                ],
            ),
            _n(
                "rd-antlers",
                "Stags shed their antlers in late winter and grow a new set through the spring.",
                [
                    ("rd-antlers-shed", True, "When do stags shed their antlers?", "Stags shed their antlers in late winter.", ()),
                    ("rd-antlers-grow", True, "When do stags grow a new set of antlers?", "Stags grow a new set through the spring.", ()),
                    ("rd-antlers-season", False, "In which season are the antlers shed?", "Stags shed their antlers in late winter.", ("winter",)),
                    ("rd-antlers-spring", False, "Do stags grow antlers through the spring?", "Stags grow a new set through the spring.", ("spring",)),
                    ("rd-antlers-late", False, "What happens to antlers in late winter?", "Stags shed their antlers in late winter.", ("shed",)),
                ],
            ),
            _n(
                "rd-food",
                "The red deer browse heather, grasses, and young shoots and are not given supplementary feed.",
                [
                    ("rd-food-browse", True, "What do the red deer browse?", "The red deer browse heather, grasses, and young shoots.", ()),
                    ("rd-food-feed", True, "Are the red deer given supplementary feed?", "The red deer are not given supplementary feed.", ("not",)),
                    ("rd-food-plants", False, "Which plants do the red deer browse?", "The red deer browse heather, grasses, and young shoots.", ("heather",)),
                    ("rd-food-extra", False, "Do the red deer get supplementary feed?", "The red deer are not given supplementary feed.", ("not",)),
                    ("rd-food-heather", False, "Do the notes list heather for the red deer?", "The red deer browse heather, grasses, and young shoots.", ("heather",)),
                ],
            ),
            _n(
                "rd-groups",
                "Hinds and calves stay in groups, while mature stags are often apart outside the rut.",
                [
                    ("rd-groups-who", True, "Which deer stay in groups?", "Hinds and calves stay in groups.", ()),
                    ("rd-groups-stags", True, "Are mature stags often apart outside the rut?", "Mature stags are often apart outside the rut.", ()),
                    ("rd-groups-hinds", False, "Do hinds stay with the calves?", "Hinds and calves stay in groups.", ("Hinds",)),
                    ("rd-groups-calves", False, "Do calves stay in groups with the hinds?", "Hinds and calves stay in groups.", ("calves",)),
                    ("rd-groups-apart", False, "Are mature stags apart outside the rut?", "Mature stags are often apart outside the rut.", ("apart",)),
                ],
            ),
            _n(
                "rd-lookout",
                "Visitors may see the deer from the marked paths and from the lookout at the Posbank.",
                [
                    ("rd-lookout-name", True, "Which lookout may visitors use to see the deer?", "Visitors may see the deer from the lookout at the Posbank.", ("Posbank",)),
                    ("rd-lookout-paths", True, "May visitors see the deer from the marked paths?", "Visitors may see the deer from the marked paths.", ()),
                    ("rd-lookout-called", False, "What lookout do the notes name for the deer?", "Visitors may see the deer from the lookout at the Posbank.", ("Posbank",)),
                    ("rd-lookout-marked", False, "Do the marked paths give a view of the deer?", "Visitors may see the deer from the marked paths.", ("marked",)),
                    ("rd-lookout-posbank", False, "Is the Posbank a lookout in the notes?", "Visitors may see the deer from the lookout at the Posbank.", ("Posbank",)),
                ],
            ),
            _n(
                "rd-shelter",
                "In hard weather the deer use the Scots pine stands for shelter.",
                [
                    ("rd-shelter-where", True, "Where do the deer go in hard weather?", "In hard weather the deer use the Scots pine stands for shelter.", ()),
                    ("rd-shelter-trees", True, "Which trees do the deer use for shelter?", "The deer use the Scots pine stands for shelter.", ("Scots",)),
                    ("rd-shelter-use", False, "Do the deer use Scots pine stands for shelter?", "The deer use the Scots pine stands for shelter.", ("Scots",)),
                    ("rd-shelter-for", False, "What do the deer use the Scots pine stands for?", "The deer use the Scots pine stands for shelter.", ("shelter",)),
                    ("rd-shelter-hard", False, "Do the deer use shelter in hard weather?", "In hard weather the deer use the Scots pine stands for shelter.", ("hard",)),
                ],
            ),
            _n(
                "rd-water",
                "The deer drink at the pools and streams on the heath.",
                [
                    ("rd-water-where", True, "Where do the deer drink?", "The deer drink at the pools and streams on the heath.", ()),
                    ("rd-water-streams", True, "Do the deer drink at streams?", "The deer drink at the pools and streams on the heath.", ("streams",)),
                    ("rd-water-pools", False, "Are there pools on the heath where the deer drink?", "The deer drink at the pools and streams on the heath.", ("pools",)),
                    ("rd-water-heath", False, "Do the deer use streams on the heath?", "The deer drink at the pools and streams on the heath.", ("heath",)),
                    ("rd-water-use", False, "What do the deer use the heath pools for?", "The deer drink at the pools and streams on the heath.", ("drink",)),
                ],
            ),
        ],
    ),
    _pack(
        "konik_oostvaardersplassen",
        "Konik horses",
        "Oostvaardersplassen",
        [
            _n("kk-species", "The Oostvaardersplassen herd is a herd of Konik horses.", []),
            _n("kk-place", "The herd lives in the Oostvaardersplassen nature reserve in Flevoland.", []),
            _n("kk-manager", "Staatsbosbeheer manages the area.", []),
            _n(
                "kk-coat",
                "Konik horses have a grey-dun coat and a dark dorsal stripe.",
                [
                    ("kk-coat-colour", True, "What coat colour do the notes give for Konik horses?", "Konik horses have a grey-dun coat.", ("grey-dun",)),
                    ("kk-coat-stripe", True, "What stripe do Konik horses have?", "Konik horses have a dark dorsal stripe.", ("dorsal",)),
                    ("kk-coat-both", False, "How do the notes describe the Konik coat?", "Konik horses have a grey-dun coat and a dark dorsal stripe.", ()),
                    ("kk-coat-dark", False, "Is the Konik dorsal stripe dark?", "Konik horses have a dark dorsal stripe.", ("dark",)),
                    ("kk-coat-dun", False, "Do the notes call the Konik coat grey-dun?", "Konik horses have a grey-dun coat.", ("grey-dun",)),
                ],
            ),
            _n(
                "kk-food",
                "The Koniks graze reeds, grasses, and willow shoots and are not given hay by visitors.",
                [
                    ("kk-food-graze", True, "What do the Koniks graze?", "The Koniks graze reeds, grasses, and willow shoots.", ()),
                    ("kk-food-hay", True, "Are the Koniks given hay by visitors?", "The Koniks are not given hay by visitors.", ("not",)),
                    ("kk-food-reeds", False, "Do the Koniks graze reeds?", "The Koniks graze reeds, grasses, and willow shoots.", ("reeds",)),
                    ("kk-food-willow", False, "Do the notes list willow shoots for the Koniks?", "The Koniks graze reeds, grasses, and willow shoots.", ("willow",)),
                    ("kk-food-extra", False, "Do visitors give the Koniks hay?", "The Koniks are not given hay by visitors.", ("not",)),
                ],
            ),
            _n(
                "kk-water",
                "The horses drink from the shallow pools and ditches in the marsh.",
                [
                    ("kk-water-where", True, "Where do the horses drink?", "The horses drink from the shallow pools and ditches in the marsh.", ()),
                    ("kk-water-ditches", True, "Do the horses drink from ditches?", "The horses drink from the shallow pools and ditches in the marsh.", ("ditches",)),
                    ("kk-water-pools", False, "Are the drinking pools shallow?", "The horses drink from the shallow pools and ditches in the marsh.", ("shallow",)),
                    ("kk-water-marsh", False, "Do the horses drink in the marsh?", "The horses drink from the shallow pools and ditches in the marsh.", ("marsh",)),
                    ("kk-water-both", False, "Do the notes mention both pools and ditches for drinking?", "The horses drink from the shallow pools and ditches in the marsh.", ("ditches",)),
                ],
            ),
            _n(
                "kk-foals",
                "Foals are usually born in spring and stay with the mare.",
                [
                    ("kk-foals-when", True, "In which season are foals usually born?", "Foals are usually born in spring.", ("spring",)),
                    ("kk-foals-mare", True, "Do foals stay with the mare?", "Foals stay with the mare.", ()),
                    ("kk-foals-both", False, "What do the notes say about foals and the mare?", "Foals are usually born in spring and stay with the mare.", ()),
                    ("kk-foals-spring", False, "Are foals usually born in spring?", "Foals are usually born in spring.", ("spring",)),
                    ("kk-foals-stay", False, "Who do the foals stay with?", "Foals stay with the mare.", ("mare",)),
                ],
            ),
            _n(
                "kk-bands",
                "The horses live in family bands led by a mare, with a stallion alongside.",
                [
                    ("kk-bands-led", True, "Who leads the family bands?", "The horses live in family bands led by a mare.", ("mare",)),
                    ("kk-bands-stallion", True, "Where is the stallion relative to the band?", "A stallion is alongside the family bands.", ()),
                    ("kk-bands-family", False, "Do the horses live in family bands?", "The horses live in family bands led by a mare, with a stallion alongside.", ()),
                    ("kk-bands-mare", False, "Is a mare the lead of the family band?", "The horses live in family bands led by a mare.", ("led",)),
                    ("kk-bands-along", False, "Do the notes place a stallion alongside the bands?", "The horses live in family bands led by a mare, with a stallion alongside.", ("stallion",)),
                ],
            ),
            _n(
                "kk-shelter",
                "In hard weather the bands use the willow scrub and reed beds for shelter.",
                [
                    ("kk-shelter-where", True, "Where do the bands shelter in hard weather?", "In hard weather the bands use the willow scrub and reed beds for shelter.", ()),
                    ("kk-shelter-reed", True, "Do the bands use reed beds for shelter?", "The bands use the willow scrub and reed beds for shelter.", ("reed",)),
                    ("kk-shelter-willow", False, "Do the bands use willow scrub for shelter?", "The bands use the willow scrub and reed beds for shelter.", ("willow",)),
                    ("kk-shelter-hard", False, "What do the bands use in hard weather?", "In hard weather the bands use the willow scrub and reed beds for shelter.", ("hard",)),
                    ("kk-shelter-both", False, "Do the notes name both willow scrub and reed beds as shelter?", "The bands use the willow scrub and reed beds for shelter.", ("scrub",)),
                ],
            ),
            _n(
                "kk-hides",
                "Visitors view the horses from the perimeter tracks and the hides.",
                [
                    ("kk-hides-where", True, "From which tracks do visitors view the horses?", "Visitors view the horses from the perimeter tracks.", ("perimeter",)),
                    ("kk-hides-hides", True, "Do visitors view the horses from the hides?", "Visitors view the horses from the hides.", ()),
                    ("kk-hides-both", False, "Where do visitors view the horses from?", "Visitors view the horses from the perimeter tracks and the hides.", ()),
                    ("kk-hides-tracks", False, "Are the viewing tracks on the perimeter?", "Visitors view the horses from the perimeter tracks.", ("perimeter",)),
                    ("kk-hides-named", False, "Do the notes mention hides for viewing the horses?", "Visitors view the horses from the hides.", ("hides",)),
                ],
            ),
        ],
    ),
    _pack(
        "beaver_biesbosch",
        "Eurasian beavers",
        "Biesbosch",
        [
            _n("bv-species", "The Biesbosch animals in these notes are Eurasian beavers.", []),
            _n("bv-place", "The colony lives in Nationaal Park De Biesbosch.", []),
            _n("bv-manager", "Staatsbosbeheer manages the national park.", []),
            _n(
                "bv-lodge",
                "Beavers build lodges of branches and mud on the water edge.",
                [
                    ("bv-lodge-what", True, "What do beavers build the lodges from?", "Beavers build lodges of branches and mud.", ()),
                    ("bv-lodge-where", True, "Where on the bank are the lodges?", "Beavers build lodges on the water edge.", ("edge",)),
                    ("bv-lodge-mud", False, "Do the lodges include mud?", "Beavers build lodges of branches and mud.", ("mud",)),
                    ("bv-lodge-branches", False, "Do the lodges include branches?", "Beavers build lodges of branches and mud.", ("branches",)),
                    ("bv-lodge-edge", False, "Are the lodges on the water edge?", "Beavers build lodges of branches and mud on the water edge.", ("edge",)),
                ],
            ),
            _n(
                "bv-dams",
                "In the narrow creeks they sometimes dam the flow with branches.",
                [
                    ("bv-dams-where", True, "In which creeks do they sometimes dam the flow?", "In the narrow creeks they sometimes dam the flow.", ("narrow",)),
                    ("bv-dams-with", True, "What do they use when they dam the flow?", "They sometimes dam the flow with branches.", ("branches",)),
                    ("bv-dams-sometimes", False, "Do they always dam the creeks?", "In the narrow creeks they sometimes dam the flow.", ("sometimes",)),
                    ("bv-dams-flow", False, "What do they dam in the narrow creeks?", "They sometimes dam the flow with branches.", ("flow",)),
                    ("bv-dams-narrow", False, "Are the dammed creeks described as narrow?", "In the narrow creeks they sometimes dam the flow.", ("narrow",)),
                ],
            ),
            _n(
                "bv-food",
                "Beavers eat bark, twigs, and aquatic plants, and they fell willow and poplar.",
                [
                    ("bv-food-eat", True, "What do beavers eat?", "Beavers eat bark, twigs, and aquatic plants.", ()),
                    ("bv-food-fell", True, "Which trees do beavers fell?", "They fell willow and poplar.", ()),
                    ("bv-food-bark", False, "Do beavers eat bark?", "Beavers eat bark, twigs, and aquatic plants.", ("bark",)),
                    ("bv-food-aquatic", False, "Do the notes list aquatic plants as beaver food?", "Beavers eat bark, twigs, and aquatic plants.", ("aquatic",)),
                    ("bv-food-poplar", False, "Do beavers fell poplar?", "They fell willow and poplar.", ("poplar",)),
                ],
            ),
            _n(
                "bv-night",
                "The beavers are mostly active at dusk and during the night.",
                [
                    ("bv-night-when", True, "When are the beavers mostly active?", "The beavers are mostly active at dusk and during the night.", ()),
                    ("bv-night-dusk", True, "Are the beavers active at dusk?", "The beavers are mostly active at dusk.", ("dusk",)),
                    ("bv-night-night", False, "Are the beavers active during the night?", "The beavers are mostly active during the night.", ("night",)),
                    ("bv-night-mostly", False, "Does activity peak at dusk and during the night?", "The beavers are mostly active at dusk and during the night.", ("mostly",)),
                    ("bv-night-both", False, "Do the notes tie beaver activity to dusk?", "The beavers are mostly active at dusk and during the night.", ("dusk",)),
                ],
            ),
            _n(
                "bv-tail",
                "A beaver tail is flat and scaly and is used as a rudder in the water.",
                [
                    ("bv-tail-shape", True, "What shape is a beaver tail?", "A beaver tail is flat and scaly.", ()),
                    ("bv-tail-use", True, "What is a beaver tail used as in the water?", "A beaver tail is used as a rudder in the water.", ("rudder",)),
                    ("bv-tail-scaly", False, "Is a beaver tail scaly?", "A beaver tail is flat and scaly.", ("scaly",)),
                    ("bv-tail-flat", False, "Is a beaver tail flat?", "A beaver tail is flat and scaly.", ("flat",)),
                    ("bv-tail-rudder", False, "Do the notes call the tail a rudder?", "A beaver tail is used as a rudder in the water.", ("rudder",)),
                ],
            ),
            _n(
                "bv-kits",
                "Kits are born inside the lodge in late spring and swim within a few weeks.",
                [
                    ("bv-kits-where", True, "Where are kits born?", "Kits are born inside the lodge.", ("lodge",)),
                    ("bv-kits-when", True, "In which season are kits born?", "Kits are born in late spring.", ("spring",)),
                    ("bv-kits-swim", False, "Do kits swim within a few weeks?", "Kits swim within a few weeks.", ("swim",)),
                    ("bv-kits-late", False, "Are kits born in late spring?", "Kits are born in late spring.", ("late",)),
                    ("bv-kits-inside", False, "Are kits born inside the lodge?", "Kits are born inside the lodge in late spring.", ("inside",)),
                ],
            ),
            _n(
                "bv-view",
                "Visitors most often notice beavers from the canoe creeks and the watch huts at dusk.",
                [
                    ("bv-view-canoe", True, "From which creeks do visitors most often notice beavers?", "Visitors most often notice beavers from the canoe creeks.", ("canoe",)),
                    ("bv-view-huts", True, "Do the notes mention watch huts for noticing beavers?", "Visitors most often notice beavers from the watch huts.", ("huts",)),
                    ("bv-view-dusk", False, "At what time do visitors most often notice beavers from the huts?", "Visitors most often notice beavers from the watch huts at dusk.", ("dusk",)),
                    ("bv-view-often", False, "Where do visitors most often notice beavers?", "Visitors most often notice beavers from the canoe creeks and the watch huts at dusk.", ()),
                    ("bv-view-both", False, "Do both canoe creeks and watch huts appear in the notes?", "Visitors most often notice beavers from the canoe creeks and the watch huts.", ("canoe",)),
                ],
            ),
        ],
    ),
    _pack(
        "highland_dwingelderveld",
        "Scottish Highland cattle",
        "Dwingelderveld",
        [
            _n("hc-species", "The Dwingelderveld grazers in these notes are Scottish Highland cattle.", []),
            _n("hc-place", "The herd lives on the heath of Nationaal Park Dwingelderveld.", []),
            _n("hc-manager", "Natuurmonumenten manages the area.", []),
            _n(
                "hc-coat",
                "Highland cattle have a long shaggy coat and wide horns.",
                [
                    ("hc-coat-coat", True, "How do the notes describe the Highland coat?", "Highland cattle have a long shaggy coat.", ("shaggy",)),
                    ("hc-coat-horns", True, "How do the notes describe the horns?", "Highland cattle have wide horns.", ("wide",)),
                    ("hc-coat-both", False, "What do the notes say about the coat and the horns?", "Highland cattle have a long shaggy coat and wide horns.", ()),
                    ("hc-coat-long", False, "Is the Highland coat long?", "Highland cattle have a long shaggy coat.", ("long",)),
                    ("hc-coat-shaggy", False, "Do the notes call the coat shaggy?", "Highland cattle have a long shaggy coat.", ("shaggy",)),
                ],
            ),
            _n(
                "hc-food",
                "Highland cattle graze heather, purple moor-grass, and birch shoots and are not fed by visitors.",
                [
                    ("hc-food-graze", True, "What do Highland cattle graze?", "Highland cattle graze heather, purple moor-grass, and birch shoots.", ()),
                    ("hc-food-fed", True, "Are Highland cattle fed by visitors?", "Highland cattle are not fed by visitors.", ("not",)),
                    ("hc-food-heather", False, "Do Highland cattle graze heather?", "Highland cattle graze heather, purple moor-grass, and birch shoots.", ("heather",)),
                    ("hc-food-birch", False, "Do the notes list birch shoots for Highland cattle?", "Highland cattle graze heather, purple moor-grass, and birch shoots.", ("birch",)),
                    ("hc-food-visitors", False, "Do visitors feed the Highland cattle?", "Highland cattle are not fed by visitors.", ("not",)),
                ],
            ),
            _n(
                "hc-winter",
                "The herd stays out on the heath through the winter.",
                [
                    ("hc-winter-where", True, "Where does the herd stay through the winter?", "The herd stays out on the heath through the winter.", ()),
                    ("hc-winter-out", True, "Does the herd stay out through the winter?", "The herd stays out on the heath through the winter.", ("out",)),
                    ("hc-winter-heath", False, "Is the winter range the heath?", "The herd stays out on the heath through the winter.", ("heath",)),
                    ("hc-winter-through", False, "Do the notes say the herd remains through the winter?", "The herd stays out on the heath through the winter.", ("winter",)),
                    ("hc-winter-stays", False, "Does the herd stay on the heath in winter?", "The herd stays out on the heath through the winter.", ("stays",)),
                ],
            ),
            _n(
                "hc-calves",
                "Calves are born in spring and have a lighter coat than the adults.",
                [
                    ("hc-calves-when", True, "In which season are calves born?", "Calves are born in spring.", ("spring",)),
                    ("hc-calves-coat", True, "How does the calf coat compare with the adults?", "Calves have a lighter coat than the adults.", ("lighter",)),
                    ("hc-calves-both", False, "What do the notes say about calves?", "Calves are born in spring and have a lighter coat than the adults.", ()),
                    ("hc-calves-spring", False, "Are calves born in spring?", "Calves are born in spring.", ("spring",)),
                    ("hc-calves-lighter", False, "Is the calf coat lighter than the adult coat?", "Calves have a lighter coat than the adults.", ("lighter",)),
                ],
            ),
            _n(
                "hc-birch",
                "The cattle keep young birch from closing over the open heath.",
                [
                    ("hc-birch-what", True, "What do the cattle keep from closing over the heath?", "The cattle keep young birch from closing over the open heath.", ()),
                    ("hc-birch-open", True, "Which heath do the cattle keep open?", "The cattle keep young birch from closing over the open heath.", ("open",)),
                    ("hc-birch-young", False, "Do the cattle hold back young birch?", "The cattle keep young birch from closing over the open heath.", ("young",)),
                    ("hc-birch-closing", False, "What are the cattle keeping from closing over?", "The cattle keep young birch from closing over the open heath.", ("closing",)),
                    ("hc-birch-keep", False, "Do the cattle keep the heath from closing over?", "The cattle keep young birch from closing over the open heath.", ("keep",)),
                ],
            ),
            _n(
                "hc-water",
                "Highland cattle drink at the peat pools and the shallow fens.",
                [
                    ("hc-water-where", True, "Where do Highland cattle drink?", "Highland cattle drink at the peat pools and the shallow fens.", ()),
                    ("hc-water-peat", True, "Do Highland cattle drink at peat pools?", "Highland cattle drink at the peat pools.", ("peat",)),
                    ("hc-water-fens", False, "Do Highland cattle drink at shallow fens?", "Highland cattle drink at the shallow fens.", ("fens",)),
                    ("hc-water-both", False, "Do the notes name both peat pools and fens?", "Highland cattle drink at the peat pools and the shallow fens.", ()),
                    ("hc-water-shallow", False, "Are the fens shallow?", "Highland cattle drink at the peat pools and the shallow fens.", ("shallow",)),
                ],
            ),
            _n(
                "hc-view",
                "Visitors see the cattle from the sheep bridges and the boardwalks across the wet heath.",
                [
                    ("hc-view-bridges", True, "From which bridges do visitors see the cattle?", "Visitors see the cattle from the sheep bridges.", ("sheep",)),
                    ("hc-view-boards", True, "Do visitors see the cattle from the boardwalks?", "Visitors see the cattle from the boardwalks across the wet heath.", ("boardwalks",)),
                    ("hc-view-both", False, "Where do visitors see the cattle from?", "Visitors see the cattle from the sheep bridges and the boardwalks across the wet heath.", ()),
                    ("hc-view-wet", False, "Are the boardwalks across wet heath?", "Visitors see the cattle from the boardwalks across the wet heath.", ("wet",)),
                    ("hc-view-sheep", False, "Do the notes mention sheep bridges?", "Visitors see the cattle from the sheep bridges.", ("sheep",)),
                ],
            ),
        ],
    ),
    _pack(
        "exmoor_lauwersmeer",
        "Exmoor ponies",
        "Lauwersmeer",
        [
            _n("ex-species", "The Lauwersmeer herd is a herd of Exmoor ponies.", []),
            _n("ex-place", "The herd lives in Nationaal Park Lauwersmeer.", []),
            _n("ex-manager", "Staatsbosbeheer manages the area.", []),
            _n(
                "ex-coat",
                "Exmoor ponies have a brown coat, a mealy muzzle, and a dark eel stripe along the back.",
                [
                    ("ex-coat-colour", True, "What coat colour do the notes give for Exmoor ponies?", "Exmoor ponies have a brown coat.", ("brown",)),
                    ("ex-coat-muzzle", True, "How do the notes describe the muzzle?", "Exmoor ponies have a mealy muzzle.", ("mealy",)),
                    ("ex-coat-stripe", False, "What stripe runs along the back?", "Exmoor ponies have a dark eel stripe along the back.", ("eel",)),
                    ("ex-coat-all", False, "How do the notes describe the Exmoor coat, muzzle, and stripe?", "Exmoor ponies have a brown coat, a mealy muzzle, and a dark eel stripe along the back.", ()),
                    ("ex-coat-dark", False, "Is the eel stripe dark?", "Exmoor ponies have a dark eel stripe along the back.", ("dark",)),
                ],
            ),
            _n(
                "ex-food",
                "The ponies graze salt marsh grasses and reeds and find their own food.",
                [
                    ("ex-food-graze", True, "What do the ponies graze?", "The ponies graze salt marsh grasses and reeds.", ()),
                    ("ex-food-own", True, "Do the ponies find their own food?", "The ponies find their own food.", ("own",)),
                    ("ex-food-reeds", False, "Do the ponies graze reeds?", "The ponies graze salt marsh grasses and reeds.", ("reeds",)),
                    ("ex-food-salt", False, "Do the notes mention salt marsh grasses?", "The ponies graze salt marsh grasses and reeds.", ("salt",)),
                    ("ex-food-both", False, "What grasses and other plants do the ponies graze?", "The ponies graze salt marsh grasses and reeds and find their own food.", ()),
                ],
            ),
            _n(
                "ex-winter",
                "Exmoor ponies keep a winter coat and stay outside through the year.",
                [
                    ("ex-winter-coat", True, "Do Exmoor ponies keep a winter coat?", "Exmoor ponies keep a winter coat.", ("winter",)),
                    ("ex-winter-out", True, "Do Exmoor ponies stay outside through the year?", "Exmoor ponies stay outside through the year.", ("outside",)),
                    ("ex-winter-both", False, "What do the notes say about winter and the rest of the year?", "Exmoor ponies keep a winter coat and stay outside through the year.", ()),
                    ("ex-winter-keep", False, "Is a winter coat kept?", "Exmoor ponies keep a winter coat.", ("keep",)),
                    ("ex-winter-year", False, "Do the ponies stay outside through the year?", "Exmoor ponies stay outside through the year.", ("year",)),
                ],
            ),
            _n(
                "ex-foals",
                "Foals are born in late spring and the band keeps them in the middle of the group.",
                [
                    ("ex-foals-when", True, "When are foals born?", "Foals are born in late spring.", ("spring",)),
                    ("ex-foals-where", True, "Where does the band keep the foals?", "The band keeps them in the middle of the group.", ("middle",)),
                    ("ex-foals-late", False, "Are foals born in late spring?", "Foals are born in late spring.", ("late",)),
                    ("ex-foals-group", False, "Do the foals stay in the middle of the group?", "The band keeps them in the middle of the group.", ("middle",)),
                    ("ex-foals-both", False, "What do the notes say about foals in the band?", "Foals are born in late spring and the band keeps them in the middle of the group.", ()),
                ],
            ),
            _n(
                "ex-bands",
                "The ponies move in small bands of mares, foals, and a stallion.",
                [
                    ("ex-bands-who", True, "Who is in the small bands?", "The ponies move in small bands of mares, foals, and a stallion.", ()),
                    ("ex-bands-stallion", True, "Do the small bands include a stallion?", "The ponies move in small bands of mares, foals, and a stallion.", ("stallion",)),
                    ("ex-bands-mares", False, "Do the bands include mares?", "The ponies move in small bands of mares, foals, and a stallion.", ("mares",)),
                    ("ex-bands-small", False, "Are the bands small?", "The ponies move in small bands of mares, foals, and a stallion.", ("small",)),
                    ("ex-bands-move", False, "Do the ponies move in bands?", "The ponies move in small bands of mares, foals, and a stallion.", ("move",)),
                ],
            ),
            _n(
                "ex-water",
                "The bands drink from the freshwater pools behind the old sea dyke.",
                [
                    ("ex-water-where", True, "Where do the bands drink?", "The bands drink from the freshwater pools behind the old sea dyke.", ()),
                    ("ex-water-fresh", True, "Are the drinking pools freshwater?", "The bands drink from the freshwater pools.", ("freshwater",)),
                    ("ex-water-dyke", False, "Which dyke are the drinking pools behind?", "The bands drink from the freshwater pools behind the old sea dyke.", ("dyke",)),
                    ("ex-water-old", False, "Is the sea dyke described as old?", "The bands drink from the freshwater pools behind the old sea dyke.", ("old",)),
                    ("ex-water-behind", False, "Are the pools behind the sea dyke?", "The bands drink from the freshwater pools behind the old sea dyke.", ("behind",)),
                ],
            ),
            _n(
                "ex-view",
                "Visitors see the ponies from the dyke path and the bird hides on the southern shore.",
                [
                    ("ex-view-path", True, "From which path do visitors see the ponies?", "Visitors see the ponies from the dyke path.", ("dyke",)),
                    ("ex-view-hides", True, "Where are the bird hides?", "The bird hides are on the southern shore.", ("southern",)),
                    ("ex-view-both", False, "Where do visitors see the ponies from?", "Visitors see the ponies from the dyke path and the bird hides on the southern shore.", ()),
                    ("ex-view-bird", False, "Do the notes mention bird hides?", "Visitors see the ponies from the bird hides on the southern shore.", ("bird",)),
                    ("ex-view-shore", False, "Is the shore in the notes the southern shore?", "The bird hides are on the southern shore.", ("southern",)),
                ],
            ),
        ],
    ),
]


# Questions over the real Kraansvlak notes. Ids match assets/dataset.json.
# Topics the ten protocol prompts already ask are not included.
WISENT_QAS: dict[str, list[dict]] = {
    "other-grazers": [
        {"id": "wisent-horses", "pair": True, "question": "Which horses graze in Kraansvlak?", "answer": "Konik horses also graze in Kraansvlak.", "must_include": ["Konik"]},
        {"id": "wisent-cattle", "pair": True, "question": "Which cattle graze in Kraansvlak?", "answer": "Scottish Highland cattle also graze in Kraansvlak.", "must_include": ["Highland"]},
        {"id": "wisent-grazers", "pair": False, "question": "Which other grazers do the notes place in Kraansvlak?", "answer": "Konik horses and Scottish Highland cattle also graze in Kraansvlak.", "must_include": []},
        {"id": "wisent-konik-yes", "pair": False, "question": "Do Konik horses graze in Kraansvlak?", "answer": "Konik horses also graze in Kraansvlak.", "must_include": ["Konik"]},
    ],
    "park": [
        {"id": "wisent-park-name", "pair": True, "question": "Which national park do the notes name?", "answer": "The herd lives in Nationaal Park Zuid-Kennemerland.", "must_include": ["Zuid-Kennemerland"]},
        {"id": "wisent-park-lives", "pair": True, "question": "What park name is written for the herd?", "answer": "The herd lives in Nationaal Park Zuid-Kennemerland.", "must_include": ["Nationaal"]},
        {"id": "wisent-park-full", "pair": False, "question": "Do the notes name Nationaal Park Zuid-Kennemerland?", "answer": "The herd lives in Nationaal Park Zuid-Kennemerland.", "must_include": ["Zuid-Kennemerland"]},
        {"id": "wisent-park-which", "pair": False, "question": "What is the park called in the notes?", "answer": "The herd lives in Nationaal Park Zuid-Kennemerland.", "must_include": ["Kennemerland"]},
    ],
    "wisentenpad": [
        {"id": "wisent-pad-open", "pair": True, "question": "Between which dates is the yellow-marked Wisentenpad open?", "answer": "The yellow-marked Wisentenpad is open from 1 September to 1 March.", "must_include": ["open", "September", "March"]},
        {"id": "wisent-pad-closed", "pair": True, "question": "When is the yellow-marked Wisentenpad closed?", "answer": "The yellow-marked Wisentenpad is closed in the bird breeding season.", "must_include": ["closed"]},
        {"id": "wisent-pad-mark", "pair": False, "question": "What colour marks the seasonal path in the notes?", "answer": "The yellow-marked Wisentenpad is open from 1 September to 1 March.", "must_include": ["yellow"]},
        {"id": "wisent-pad-breed", "pair": False, "question": "Is the yellow-marked Wisentenpad open in the bird breeding season?", "answer": "The yellow-marked Wisentenpad is closed in the bird breeding season.", "must_include": ["closed"]},
    ],
    "no-feeding": [
        {"id": "wisent-feed-people", "pair": True, "question": "Are the wisents fed by people?", "answer": "The wisents are not fed by people.", "must_include": ["not"]},
        {"id": "wisent-feed-where", "pair": True, "question": "Where do the wisents find their food?", "answer": "The wisents find their own food in the dunes.", "must_include": ["dunes"]},
        {"id": "wisent-feed-do", "pair": False, "question": "Do people feed the wisents?", "answer": "The wisents are not fed by people.", "must_include": ["not"]},
        {"id": "wisent-feed-dunes", "pair": False, "question": "Do the wisents find food in the dunes?", "answer": "The wisents find their own food in the dunes.", "must_include": ["dunes"]},
    ],
}
