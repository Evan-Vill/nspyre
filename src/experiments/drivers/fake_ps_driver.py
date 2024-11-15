class FakePSDriver:
    """Simulate a device driver for a power supply."""
    def __init__(self):
        self.channels = list(range(12))

    def set_voltage(self, channel: int, val: float):
        if channel not in self.channels:
            raise ValueError(f'Given channel [{channel}] must be an int in range [0, {len(self.channels)})')
        self.channels[channel] = val

    def get_voltage(self, channel: int) -> float:
        if channel not in self.channels:
            raise ValueError(f'Given channel [{channel}] must be an int in range [0, {len(self.channels)})')
        return self.channels[channel]
