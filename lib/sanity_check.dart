/// Fixed prompts run after every model load. Neither is a protocol prompt.
const sanityMaxTokens = 32;

const sanityAnimalPrompt = 'Name one European wild animal in three words.';
const sanityMathPrompt = 'What is 2+2?';

class SanityPrompt {
  const SanityPrompt({required this.text, required this.requireDigitFour});

  final String text;
  final bool requireDigitFour;
}

const sanityPrompts = <SanityPrompt>[
  SanityPrompt(text: sanityAnimalPrompt, requireDigitFour: false),
  SanityPrompt(text: sanityMathPrompt, requireDigitFour: true),
];

/// Why a sanity answer failed, or null when it passes.
///
/// A pass is a short answer that is not token soup. The math prompt must
/// also contain the digit 4. Hitting [maxTokens] is a failure on its own:
/// the broken phone runs ran until the cap.
String? sanityFailureReason({
  required String output,
  required bool hitMaxTokens,
  required bool requireDigitFour,
}) {
  final trimmed = output.trim();
  if (trimmed.isEmpty) return 'empty';
  if (hitMaxTokens) return 'hit max tokens';
  if (looksLikeTokenSoup(trimmed)) return 'token soup';
  if (requireDigitFour && !trimmed.contains('4')) {
    return 'answer does not contain 4';
  }
  return null;
}

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
