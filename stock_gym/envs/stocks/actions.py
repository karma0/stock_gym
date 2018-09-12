"""Environment for trading"""

from collections import namedtuple


State = namedtuple('State', 'buy sell stay')


class ExchangeAction:
    """The actions of the market"""
    state = State(False, False, True)  # buy, sell, stay
    actions = ["buy", "sell", "stay"]

    def __init__(self, action=None):
        self.reset(action)

    def reset(self, action=None):
        """Reset the action state"""
        if action is not None:
            assert action in self.actions
            if action == "buy":
                self.state = State(True, False, False)
            elif action == "sell":
                self.state = State(False, True, False)
            else:
                self.state = State(False, False, True)
        else:
            self.state = State(False, False, True)
        return self.state

    def changed(self, action):
        """Test, returning True if state changed"""
        return True if [a for a in self.actions if action == a] else False
