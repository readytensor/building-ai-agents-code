from eval.sampling import filter_pool
from eval.targets.swebench import to_instance
from eval.tests.test_swebench_provider import FAKE_RECORD


def _inst(id, difficulty, repo):
    class Stub:
        pass
    i = Stub()
    i.id = id
    i.meta = {"difficulty": difficulty, "repo": repo}
    return i


POOL = [
    _inst("a", "<15 min fix", "pallets/flask"),
    _inst("b", "15 min - 1 hour", "psf/requests"),
    _inst("c", "1-4 hours", "django/django"),
    _inst("d", ">4 hours", "django/django"),
]


def test_difficulty_buckets_map_to_dataset_labels():
    assert [i.id for i in filter_pool(POOL, difficulty="easy")] == ["a"]
    assert [i.id for i in filter_pool(POOL, difficulty="medium")] == ["b"]
    assert [i.id for i in filter_pool(POOL, difficulty="hard")] == ["c", "d"]


def test_repo_filter_is_substring_match():
    assert [i.id for i in filter_pool(POOL, repo="django")] == ["c", "d"]
    assert [i.id for i in filter_pool(POOL, repo="flask")] == ["a"]


def test_filters_combine():
    assert [i.id for i in filter_pool(POOL, difficulty="hard", repo="django")] == [
        "c", "d"]
    assert filter_pool(POOL, difficulty="easy", repo="django") == []


def test_no_filters_returns_pool_unchanged():
    assert filter_pool(POOL) == POOL


def test_swebench_instance_carries_meta():
    inst = to_instance(FAKE_RECORD)
    assert inst.meta == {"difficulty": "<15 min fix", "repo": "demo/demo"}
