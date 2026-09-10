from agent.evaluator import check_empty_answer, check_named_entity_overlap


def test_empty_answer_true():
    assert check_empty_answer("") is True
    assert check_empty_answer("抱歉，没有找到") is True


def test_empty_answer_false():
    assert check_empty_answer("这是正常的回答内容。") is False


def test_overlap_full():
    answer = "对应装备是【炎尾龙头盔】"
    docs = ["【炎尾龙头盔】需要素材A"]
    assert check_named_entity_overlap(answer, docs) == 1.0


def test_overlap_partial():
    answer = "【炎尾龙头盔】和【不存在的装备】"
    docs = ["【炎尾龙头盔】需要素材A"]
    assert check_named_entity_overlap(answer, docs) == 0.5


def test_no_entities_skips():
    answer = "没有引用任何方括号实体"
    docs = ["内容"]
    assert check_named_entity_overlap(answer, docs) == 1.0
