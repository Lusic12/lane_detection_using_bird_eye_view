import spidev

class SPI:
    def __init__(self, bus=0, device=0, speed_hz=1000000):
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = speed_hz

    def transfer(self, data):
        return self.spi.xfer2(data)

    def close(self):
        self.spi.close()
