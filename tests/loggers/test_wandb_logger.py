import unittest


class TestWandbLogger(unittest.TestCase):
    """Test WandbLogger."""
    def setUp(self) -> None:
        self.workspace = "uetai_tester"
        self.api_key = "Qd9kYrmr6gq4ouD4GG9TvuxJ6"

    def test_logger_init(self):
        self.assertEqual(True, False)  # add assertion here

    def test_logger_function(self):
        pass


if __name__ == '__main__':
    unittest.main()
