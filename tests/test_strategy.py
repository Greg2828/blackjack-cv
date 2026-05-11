from src.game.card import Card
from src.game.hand import Hand
from src.game.state import Action
from src.decision.strategy import recommend

# Helpers
def hand(*ranks: str) -> Hand:
    return Hand([Card(r) for r in ranks])

def upcard(rank: str) -> Card:
    return Card(rank)


# --- Pares ---

def test_pair_aces_siempre_split():
    h = hand('A', 'A')
    for rank in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']:
        assert recommend(h, upcard(rank)) == Action.SPLIT

def test_pair_ocho_siempre_split():
    h = hand('8', '8')
    for rank in ['2', '3', '5', '7', '9', '10', 'A']:
        assert recommend(h, upcard(rank)) == Action.SPLIT

def test_par_dieces_siempre_stand():
    for rank_pair in ['10', 'J', 'Q', 'K']:
        h = hand(rank_pair, rank_pair)
        for up in ['5', '6', '7']:
            assert recommend(h, upcard(up)) == Action.STAND

def test_par_cincos_no_split_sino_double():
    h = hand('5', '5')
    # 5+5=10, trata como hard 10: doblar vs 2-9
    assert recommend(h, upcard('6')) == Action.DOUBLE
    assert recommend(h, upcard('9')) == Action.DOUBLE
    assert recommend(h, upcard('10')) == Action.HIT


# --- Soft hands ---

def test_soft_18_dobla_vs_dealer_3_a_6():
    h = hand('A', '7')  # soft 18
    for up in ['3', '4', '5', '6']:
        assert recommend(h, upcard(up)) == Action.DOUBLE

def test_soft_18_stand_vs_dealer_7_8():
    h = hand('A', '7')
    for up in ['7', '8']:
        assert recommend(h, upcard(up)) == Action.STAND

def test_soft_18_hit_vs_dealer_9_10_A():
    h = hand('A', '7')
    for up in ['9', '10', 'A']:
        assert recommend(h, upcard(up)) == Action.HIT

def test_soft_17_dobla_vs_dealer_3_a_6():
    h = hand('A', '6')
    for up in ['3', '4', '5', '6']:
        assert recommend(h, upcard(up)) == Action.DOUBLE

def test_soft_sin_double_usa_fallback():
    h = hand('A', '7')  # soft 18, doble vs 3-6 → fallback stand
    assert recommend(h, upcard('5'), can_double=False) == Action.STAND

    h2 = hand('A', '6')  # soft 17, doble vs 3-6 → fallback hit
    assert recommend(h2, upcard('4'), can_double=False) == Action.HIT


# --- Hard hands ---

def test_hard_11_dobla_vs_2_a_10():
    h = hand('6', '5')
    for up in ['2', '3', '4', '5', '6', '7', '8', '9', '10']:
        assert recommend(h, upcard(up)) == Action.DOUBLE

def test_hard_11_hit_vs_as():
    h = hand('6', '5')
    assert recommend(h, upcard('A')) == Action.HIT

def test_hard_16_surrender_vs_9_10_A():
    h = hand('9', '7')
    assert recommend(h, upcard('9'))  == Action.SURRENDER
    assert recommend(h, upcard('10')) == Action.SURRENDER
    assert recommend(h, upcard('A'))  == Action.SURRENDER

def test_hard_16_stand_vs_dealer_2_a_6():
    h = hand('9', '7')
    for up in ['2', '3', '4', '5', '6']:
        assert recommend(h, upcard(up)) == Action.STAND

def test_hard_15_surrender_vs_10():
    h = hand('9', '6')
    assert recommend(h, upcard('10')) == Action.SURRENDER

def test_hard_17_plus_siempre_stand():
    for total_rank in [('10', '7'), ('10', '8'), ('10', 'J')]:
        h = hand(*total_rank)
        for up in ['2', '5', '7', '10', 'A']:
            assert recommend(h, upcard(up)) == Action.STAND

def test_hard_12_stand_vs_4_5_6():
    h = hand('10', '2')
    for up in ['4', '5', '6']:
        assert recommend(h, upcard(up)) == Action.STAND

def test_hard_8_hit_siempre():
    h = hand('3', '5')
    for up in ['2', '6', '10', 'A']:
        assert recommend(h, upcard(up)) == Action.HIT


# --- Sin surrender disponible ---

def test_no_surrender_fallback_a_hit():
    h = hand('9', '7')  # hard 16 vs 10 → surrender, fallback hit
    assert recommend(h, upcard('10'), can_surrender=False) == Action.HIT
