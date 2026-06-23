from training_lock import ExclusionLock


def test_mutual_exclusion():
    lk = ExclusionLock()
    assert lk.try_acquire("training") is True
    assert lk.is_held() is True
    assert lk.try_acquire("generation") is False  # blocked
    lk.release("training")
    assert lk.is_held() is False
    assert lk.try_acquire("generation") is True


def test_release_by_wrong_holder_is_noop():
    lk = ExclusionLock()
    lk.try_acquire("training")
    lk.release("generation")  # not the holder
    assert lk.current_holder() == "training"
