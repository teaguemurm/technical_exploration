import 'package:flutter_test/flutter_test.dart';
import 'package:technical_exploration/sanity_check.dart';

void main() {
  test('english answers pass and Fold-style soup fails', () {
    expect(
      sanityFailureReason(
        output: 'European bison.',
        hitMaxTokens: false,
        requireDigitFour: false,
        requireAnimalWord: true,
      ),
      isNull,
    );
    expect(
      sanityFailureReason(
        output: '4',
        hitMaxTokens: false,
        requireDigitFour: true,
        requireAnimalWord: false,
      ),
      isNull,
    );
    expect(
      sanityFailureReason(
        output: '2 + 2 = 4',
        hitMaxTokens: false,
        requireDigitFour: true,
        requireAnimalWord: false,
      ),
      isNull,
    );
    expect(
      sanityFailureReason(
        output: '',
        hitMaxTokens: false,
        requireDigitFour: false,
        requireAnimalWord: false,
      ),
      'empty',
    );
    expect(
      sanityFailureReason(
        output: 'The herd is wisents.',
        hitMaxTokens: true,
        requireDigitFour: false,
        requireAnimalWord: true,
      ),
      'hit max tokens',
    );
    expect(
      sanityFailureReason(
        output: 'I do not know',
        hitMaxTokens: false,
        requireDigitFour: true,
        requireAnimalWord: false,
      ),
      'answer does not contain 4',
    );

    const foldSoup = 'Tulthingbiosaments话atrix Davies.rs 应用查看下一篇下一篇学问пед淹';
    expect(looksLikeTokenSoup(foldSoup), isTrue);
    expect(
      sanityFailureReason(
        output: foldSoup,
        hitMaxTokens: false,
        requireDigitFour: false,
        requireAnimalWord: false,
      ),
      'token soup',
    );
    expect(
      sanityFailureReason(
        output: 'Quod erat demonstrandum lorem ipsum.',
        hitMaxTokens: false,
        requireDigitFour: false,
        requireAnimalWord: true,
      ),
      'answer does not name an animal',
    );
    expect(looksLikeTokenSoup('Quod erat demonstrandum lorem ipsum.'), isFalse);
    expect(
      containsRecognisableAnimalWord('Tulthingbiosamentsatrix Davies'),
      isFalse,
    );
    expect(containsRecognisableAnimalWord('A white stork.'), isTrue);
    expect(containsRecognisableAnimalWord('Fire salamander.'), isTrue);
    expect(containsRecognisableAnimalWord('Common viper.'), isTrue);
  });

  test('both fixed prompts are required', () {
    expect(sanityPrompts, hasLength(2));
    expect(sanityPrompts[0].text, sanityAnimalPrompt);
    expect(sanityPrompts[0].requireDigitFour, isFalse);
    expect(sanityPrompts[0].requireAnimalWord, isTrue);
    expect(sanityPrompts[1].text, sanityMathPrompt);
    expect(sanityPrompts[1].requireDigitFour, isTrue);
    expect(sanityPrompts[1].requireAnimalWord, isFalse);
    expect(sanityMaxTokens, 32);
  });
}
