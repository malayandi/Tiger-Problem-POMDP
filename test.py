import unittest

from game import Game

class tigerTest(unittest.TestCase):

    def testStateAfterListen(self):
        game = Game()
        # testing 10 times to account for uncertainty
        for i in range(10):
            old_state = game.getState()
            game.respond("listen")
            new_state = game.getState()
            self.assertFalse(old_state != new_state)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
