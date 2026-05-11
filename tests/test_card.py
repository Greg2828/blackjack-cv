import pytest
from src.game.card import Card


def test_number_card_value():
    assert Card('7').value == 7


def test_face_cards_are_10():
    assert Card('J').value == 10
    assert Card('Q').value == 10
    assert Card('K').value == 10


def test_ace_is_11():
    assert Card('A').value == 11


def test_back_has_no_value():
    with pytest.raises(ValueError):
        Card('BACK').value


def test_is_ace():
    assert Card('A').is_ace
    assert not Card('K').is_ace


def test_is_back():
    assert Card('BACK').is_back
    assert not Card('A').is_back
