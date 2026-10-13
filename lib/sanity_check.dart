/// Fixed prompts run after every model load. Neither is a protocol prompt.
const sanityMaxTokens = 32;

const sanityAnimalPrompt = 'Name one European wild animal in three words.';
const sanityMathPrompt = 'What is 2+2?';

class SanityPrompt {
  const SanityPrompt({
    required this.text,
    required this.requireDigitFour,
    required this.requireAnimalWord,
  });

  final String text;
  final bool requireDigitFour;
  final bool requireAnimalWord;
}

const sanityPrompts = <SanityPrompt>[
  SanityPrompt(
    text: sanityAnimalPrompt,
    requireDigitFour: false,
    requireAnimalWord: true,
  ),
  SanityPrompt(
    text: sanityMathPrompt,
    requireDigitFour: true,
    requireAnimalWord: false,
  ),
];

/// Why a sanity answer failed, or null when it passes.
///
/// A pass is a short answer that is not token soup. The math prompt must
/// also contain the digit 4. The animal prompt must contain a recognisable
/// animal word: some 12 Oct phone output was Latin-script soup, which the
/// mixed-script check does not catch. Hitting [maxTokens] is a failure on
/// its own: the broken phone runs ran until the cap.
String? sanityFailureReason({
  required String output,
  required bool hitMaxTokens,
  required bool requireDigitFour,
  required bool requireAnimalWord,
}) {
  final trimmed = output.trim();
  if (trimmed.isEmpty) return 'empty';
  if (hitMaxTokens) return 'hit max tokens';
  if (looksLikeTokenSoup(trimmed)) return 'token soup';
  if (requireAnimalWord && !containsRecognisableAnimalWord(trimmed)) {
    return 'answer does not name an animal';
  }
  if (requireDigitFour && !trimmed.contains('4')) {
    return 'answer does not contain 4';
  }
  return null;
}

/// Whole-word match against [recognisableAnimalWords].
///
/// "bison." matches. "biosaments" does not. Matching is case-insensitive.
bool containsRecognisableAnimalWord(String text) {
  for (final match in RegExp(r'[A-Za-z]+').allMatches(text)) {
    if (recognisableAnimalWords.contains(match.group(0)!.toLowerCase())) {
      return true;
    }
  }
  return false;
}

/// European fauna a short correct answer might name, including plurals.
///
/// Mammals, birds, reptiles, and amphibians. The check exists so a coherent
/// animal name passes and Latin-script soup fails, not so the species has to
/// be wild rather than domestic.
const recognisableAnimalWords = <String>{
  'adder',
  'adders',
  'albatross',
  'asp',
  'asps',
  'aurochs',
  'avocet',
  'avocets',
  'badger',
  'badgers',
  'bat',
  'bats',
  'bear',
  'bears',
  'beaver',
  'beavers',
  'bison',
  'bisons',
  'bittern',
  'bitterns',
  'blackbird',
  'blackbirds',
  'boar',
  'boars',
  'buzzard',
  'buzzards',
  'capercaillie',
  'cat',
  'cats',
  'chamois',
  'chough',
  'choughs',
  'coot',
  'coots',
  'cormorant',
  'cormorants',
  'crane',
  'cranes',
  'crow',
  'crows',
  'cuckoo',
  'cuckoos',
  'curlew',
  'curlews',
  'deer',
  'dolphin',
  'dolphins',
  'dormice',
  'dormouse',
  'dove',
  'doves',
  'duck',
  'ducks',
  'eagle',
  'eagles',
  'egret',
  'egrets',
  'elk',
  'elks',
  'falcon',
  'falcons',
  'finch',
  'finches',
  'flamingo',
  'flamingos',
  'fox',
  'foxes',
  'frog',
  'froglet',
  'frogs',
  'gannet',
  'gannets',
  'gecko',
  'geckos',
  'geese',
  'goose',
  'goshawk',
  'goshawks',
  'grebe',
  'grebes',
  'grouse',
  'gull',
  'gulls',
  'hamster',
  'hamsters',
  'hare',
  'hares',
  'harrier',
  'harriers',
  'hawk',
  'hawks',
  'hedgehog',
  'hedgehogs',
  'heron',
  'herons',
  'hoopoe',
  'ibex',
  'ibexes',
  'jackdaw',
  'jackdaws',
  'jay',
  'jays',
  'kestrel',
  'kestrels',
  'kingfisher',
  'kingfishers',
  'kite',
  'kites',
  'lapwing',
  'lapwings',
  'lark',
  'larks',
  'lizard',
  'lizards',
  'lynx',
  'lynxes',
  'magpie',
  'magpies',
  'marmot',
  'marmots',
  'marten',
  'martens',
  'mice',
  'mole',
  'moles',
  'moorhen',
  'moorhens',
  'moose',
  'mouflon',
  'mouflons',
  'mouse',
  'newt',
  'newts',
  'nightingale',
  'nightingales',
  'nuthatch',
  'nuthatches',
  'oriole',
  'orioles',
  'osprey',
  'ospreys',
  'otter',
  'otters',
  'owl',
  'owlet',
  'owls',
  'oystercatcher',
  'oystercatchers',
  'partridge',
  'partridges',
  'pelican',
  'pelicans',
  'petrel',
  'petrels',
  'pheasant',
  'pheasants',
  'pigeon',
  'pigeons',
  'plover',
  'plovers',
  'polecat',
  'polecats',
  'porpoise',
  'porpoises',
  'ptarmigan',
  'ptarmigans',
  'puffin',
  'puffins',
  'quail',
  'quails',
  'rabbit',
  'rabbits',
  'rat',
  'rats',
  'raven',
  'ravens',
  'reindeer',
  'robin',
  'robins',
  'roller',
  'rollers',
  'salamander',
  'salamanders',
  'seal',
  'seals',
  'shearwater',
  'shearwaters',
  'shrew',
  'shrews',
  'shrike',
  'shrikes',
  'skink',
  'skinks',
  'slowworm',
  'slowworms',
  'snake',
  'snakes',
  'snipe',
  'sparrow',
  'sparrowhawk',
  'sparrowhawks',
  'sparrows',
  'spoonbill',
  'spoonbills',
  'squirrel',
  'squirrels',
  'stag',
  'stags',
  'starling',
  'starlings',
  'stoat',
  'stoats',
  'stork',
  'storks',
  'swallow',
  'swallows',
  'swan',
  'swans',
  'swift',
  'swifts',
  'tadpole',
  'tadpoles',
  'tern',
  'terns',
  'thrush',
  'thrushes',
  'tit',
  'titmouse',
  'tits',
  'toad',
  'toads',
  'tortoise',
  'tortoises',
  'treecreeper',
  'treecreepers',
  'turtle',
  'turtles',
  'viper',
  'vipers',
  'vole',
  'voles',
  'vulture',
  'vultures',
  'walrus',
  'walruses',
  'warbler',
  'warblers',
  'weasel',
  'weasels',
  'whale',
  'whales',
  'wildcat',
  'wildcats',
  'wisent',
  'wisents',
  'wolf',
  'wolverine',
  'wolverines',
  'wolves',
  'woodcock',
  'woodcocks',
  'woodpecker',
  'woodpeckers',
  'wren',
  'wrens',
};

/// Mixed-script gibberish from a wrong forward pass.
///
/// Latin-1 and Latin Extended count as Latin, so an English sentence passes.
/// A handful of CJK, Cyrillic, Arabic, or Thai letters fails, which is the
/// shape of the Fold 8 outputs. A numeric answer such as "4" is not soup.
bool looksLikeTokenSoup(String text) {
  var letters = 0;
  var latin = 0;
  var nonLatin = 0;
  for (final rune in text.runes) {
    if (!_isLetter(rune)) continue;
    letters++;
    if (_isLatinLetter(rune)) {
      latin++;
    } else {
      nonLatin++;
    }
  }
  if (nonLatin >= 4) return true;
  if (letters >= 12 && latin / letters < 0.6) return true;
  return false;
}

bool _isLatinLetter(int rune) =>
    (rune >= 0x41 && rune <= 0x5A) ||
    (rune >= 0x61 && rune <= 0x7A) ||
    (rune >= 0xC0 && rune <= 0x24F);

bool _isLetter(int rune) {
  if (_isLatinLetter(rune)) return true;
  if (rune >= 0x370 && rune <= 0x1FFF) return true;
  if (rune >= 0x2E80 && rune <= 0x9FFF) return true;
  if (rune >= 0xAC00 && rune <= 0xD7AF) return true;
  if (rune >= 0x0600 && rune <= 0x06FF) return true;
  if (rune >= 0x0E00 && rune <= 0x0E7F) return true;
  return false;
}
