from detector.preprocessing import clean_text, word_count


def test_strips_reuters_dateline():
    assert clean_text("WASHINGTON (Reuters) - The Senate voted.") == "the senate voted"


def test_strips_multi_city_dateline():
    out = clean_text("WEST PALM BEACH, Fla./WASHINGTON (Reuters) - Talks begin.")
    assert out == "talks begin"


def test_strips_trailing_image_credit():
    out = clean_text("He said it was great.Featured image via Getty Images")
    assert out == "he said it was great"


def test_apostrophes_are_normalised_the_same_way():
    # The dataset mixes curly, straight and missing apostrophes.
    assert clean_text("couldn’t") == clean_text("couldn't") == clean_text("couldn t") == "couldnt"
    assert clean_text("it s over, we ve won") == "its over weve won"


def test_removes_urls_handles_and_punctuation():
    out = clean_text("Read more at https://example.com/x @someone #hashtag!!! 2017")
    assert out == "read more at"


def test_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_word_count():
    assert word_count("  one two   three ") == 3
