"""Environment for trading"""


class ExchangeAction:
    """The actions of the market"""
    long = False  # By default we're not long on a position
    actions = [False, True]  # short, long

    def __init__(self, action=None):
        self.reset(action)

    def reset(self, action=None):
        """Reset the action state"""
        if action is not None:
            assert action in self.actions
            self.long = action
        else:
            self.long = False  # default back to short
        return self.long

    def changed(self, action):
        """Test, returning True if state changed"""
        return action is not self.long
