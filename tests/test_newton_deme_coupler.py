import unittest

from mophi.couplers.newton_deme import NewtonDEMEContactCoupler, NewtonDEMEOwnerMap


class NewtonDEMEOwnerMapTest(unittest.TestCase):
    def test_accepts_matching_consecutive_owner_span(self):
        owner_map = NewtonDEMEOwnerMap([7, 2, 11], [40, 41, 42])

        self.assertEqual(owner_map.newton_body_indices, (7, 2, 11))
        self.assertEqual(owner_map.deme_owner_ids, (40, 41, 42))
        self.assertEqual(owner_map.first_deme_owner_id, 40)
        self.assertEqual(owner_map.owner_count, 3)

    def test_rejects_empty_mapping(self):
        with self.assertRaisesRegex(ValueError, "at least one body"):
            NewtonDEMEOwnerMap([], [])

    def test_rejects_different_mapping_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            NewtonDEMEOwnerMap([0, 1], [20])

    def test_rejects_duplicate_newton_body(self):
        with self.assertRaisesRegex(ValueError, "only once"):
            NewtonDEMEOwnerMap([3, 3], [20, 21])

    def test_rejects_nonconsecutive_deme_owners(self):
        with self.assertRaisesRegex(ValueError, "consecutive span"):
            NewtonDEMEOwnerMap([3, 4], [20, 22])


class NewtonDEMEContactCouplerValidationTest(unittest.TestCase):
    def test_rejects_non_cuda_device_before_allocating_buffers(self):
        class Model:
            body_count = 1

        class Device:
            is_cuda = False

        with self.assertRaisesRegex(ValueError, "requires a CUDA"):
            coupler = NewtonDEMEContactCoupler()
            coupler.initialize(Model(), object(), NewtonDEMEOwnerMap([0], [0]), Device())

    def test_methods_require_initialize(self):
        coupler = NewtonDEMEContactCoupler()

        with self.assertRaisesRegex(RuntimeError, "must be initialized"):
            coupler.step_deme()


if __name__ == "__main__":
    unittest.main()
