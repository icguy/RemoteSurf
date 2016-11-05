from pyModbusTCP.client import ModbusClient
import time

SERVER_HOST = "192.168.0.104"
SERVER_PORT = 502

def intToUint16(val):
    assert -32768 <= val <= 32767
    return val if val >= 0 else 65536 + val

class ModbusClientRS:
    def __init__(self):
        self.client = ModbusClient()

    def writeRegister(self, address, value):
        if self.client.is_open():
            return self.client.write_single_register(address, value)
        return None

    def readRegister(self, address, value):
        if self.client.is_open():
            self.client.read_holding_registers(address, value)

    def connect(self, host, port):
        # self.client.debug(True)
        self.client.host(SERVER_HOST)
        self.client.port(SERVER_PORT)

        if not self.client.is_open():
            if not self.client.open():
                print("unable to connect to " + SERVER_HOST + ":" + str(SERVER_PORT))

    def is_open(self):
        return self.client.is_open()

    def disconnect(self):
        return self.client.close()


def test():
    c = ModbusClient()
    # uncomment this line to see debug message
    # c.debug(True)
    # define modbus server host, port
    c.host(SERVER_HOST)
    c.port(SERVER_PORT)
    while True:
        # open or reconnect TCP to server
        if not c.is_open():
            if not c.open():
                print("unable to connect to " + SERVER_HOST + ":" + str(SERVER_PORT))

        # if open() is ok, read register (modbus function 0x03)
        if c.is_open():
            print c.write_single_register(504, intToUint16(-216))

        # sleep 2s before next polling
        time.sleep(2)

if __name__ == '__main__':
    test()