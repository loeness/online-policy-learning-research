import numpy as np

from scripts.run_knowledge_revisit_corrected import apply_allocation, episode_plan


def test_allocation_clears_only_reassigned_slot_and_preserves_shared_head():
    matrix = np.ones((6, 27), dtype=np.float32)
    assert apply_allocation(matrix, {"allocated_slot": 2}, shared=False)
    assert not matrix[2].any()
    assert matrix[[0, 1, 3, 4, 5]].all()
    shared = np.ones_like(matrix)
    assert not apply_allocation(shared, {"allocated_slot": 0}, shared=True)
    assert shared.all()


def test_episode_random_plan_is_independent_of_previous_episode_length():
    reset_a, action_a, uniforms_a = episode_plan(401, 2, 1)
    # A short preceding episode cannot consume the next episode's stream.
    episode_plan(401, 2, 0)[2][:7]
    reset_b, action_b, uniforms_b = episode_plan(401, 2, 1)
    assert (reset_a, action_a) == (reset_b, action_b)
    assert np.array_equal(uniforms_a, uniforms_b)
    assert not np.array_equal(uniforms_a, episode_plan(401, 2, 0)[2])
