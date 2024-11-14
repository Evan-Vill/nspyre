class FakeIODriver:
    """Simulate a device driver for digital IO."""
    def __init__(self):
        self.channels = list(range(12))

    def set_out(self, channel: int, val: bool):
        if channel not in self.channels:
            raise ValueError(f'Given channel [{channel}] must be an int in range [0, {len(self.channels)})')
        self.channels[channel] = val

    def read(self, channel: int) -> bool:
        if channel not in self.channels:
            raise ValueError(f'Given channel [{channel}] must be an int in range [0, {len(self.channels)})')
        return self.channels[channel]
