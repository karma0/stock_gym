"""Environment for trading"""

from collections import namedtuple


State = namedtuple('State', 'long short stay')


class ExchangeAction:
    """The actions of the market"""
    state = State(False, False, True)  # long, short, stay
    actions = ["long", "short", "stay"]

    def __init__(self, action=None):
        self.reset(action)

    def reset(self, action=None):
        """Reset the action state"""
        if action is not None:
            assert action in self.actions
            if action == "long":
                self.state = State(True, False, False)
            elif action == "long":
                self.state = State(False, True, False)
            else:
                self.state = State(False, False, True)
        else:
            self.state = State(False, False, True)
        return self.state

    def changed(self, action):
        """Test, returning True if state changed"""
        return True if [a for a in self.actions if action == a] else False
